#######  Set up spark session ##############
import pyarrow as pa
import pyarrow.parquet as pq
from pyspark.sql import SparkSession
import pyspark.sql.functions as f
from pyspark import SparkConf
from pyspark.sql.window import Window
spark.sql("set spark.sql.legacy.timeParserPolicy=LEGACY")
spark.sql("set spark.sql.autoBroadcastJoinThreshold = -1")



print("loaded libraries")
spark = SparkSession.builder \
        .appName("worst vitals") \
        .getOrCreate()

## Read in Vitals
vitals = spark.read.parquet("/project2/wparker/SIPA_data/RCLIF_vitals.parquet")

vitals = vitals.withColumn('meas_hour', f.hour(f.col('recorded_time')))
vitals = vitals.withColumn('meas_date', f.to_date(f.col('recorded_time')))

vitals = vitals.filter(f.col('vital_name').isNotNull())
vitals = vitals.filter(f.col('vital_value').isNotNull())
vitals = vitals.filter(f.col('vital_value')>0)

## Calculate MAP using sbp and dbp taken at same time, before hourly blocking
map = vitals.filter((f.col('vital_name')=='sbp') |
                    (f.col('vital_name')=='dbp') |
                    (f.col('vital_name')=='MAP'))
map = map.filter(f.col('vital_value')>=30)

group_cols = ["encounter_id",'recorded_time']
map_wide = map.groupBy(group_cols) \
                    .pivot("vital_name").agg((f.first("vital_value").alias("value")))

map_wide = map_wide.withColumn("MAP_combined", f.expr(
        """
        CASE
        WHEN MAP IS NOT NULL AND MAP <250 THEN MAP
        WHEN MAP IS NULL AND sbp IS NOT NULL AND sbp < 250 AND dbp < 200 AND dbp IS NOT NULL THEN ( sbp + (2.0 * dbp )) / 3.0
        ELSE NULL
        END
        """
    ))
map_wide.select('encounter_id','recorded_time', 'MAP_combined')
map_wide = map_wide.withColumn('meas_hour', f.hour(f.col('recorded_time')))
map_wide = map_wide.withColumn('meas_date', f.to_date(f.col('recorded_time')))

group_cols = ["encounter_id",'meas_date', 'meas_hour']
map_wide_grouped = map_wide.groupBy(group_cols) \
                    .agg(f.min('MAP_combined').alias("MAP_combined_min"),
                    f.max('MAP_combined').alias("MAP_combined_max")).orderBy(group_cols)



vitals_wide = vitals.groupBy(group_cols) \
                    .pivot("vital_name") \
                    .agg(f.min('vital_value').alias("min"),
                    f.max('vital_value').alias("max")).orderBy(group_cols)

vitals_wide = vitals_wide.join(map_wide_grouped, on=group_cols, how='full')

## Join to cohort hourly blocked dataset
cohort_hours = spark.read.parquet("/project2/wparker/SIPA_data/life_support_cohort_48.parquet")

group_cols = ["encounter_id", 'meas_date', 'meas_hour']
cohort_vitals_48 = cohort_hours.join(vitals_wide, on=group_cols, how='left')


cohort_vitals_48.write.parquet("/project2/wparker/SIPA_data/cohort_vitals_48.parquet", mode="overwrite")





