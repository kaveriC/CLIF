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
        .appName("worst labs") \
        .getOrCreate()

## Read in Labs
labs = spark.read.parquet("/project2/wparker/SIPA_data/RCLIF_labs.parquet")

select_expr = [f.regexp_replace(f.col('lab_name'), "[\ufeff]", "").alias('lab_name')]
labs = labs.select('encounter_id', 'lab_result_time', 'lab_value', *select_expr)

select_expr = [f.regexp_replace(f.col('lab_value'), "[<]", "").alias('lab_value')]
labs = labs.select('encounter_id', 'lab_result_time', 'lab_name', *select_expr)

select_expr = [f.regexp_replace(f.col('lab_value'), "[>]", "").alias('lab_value')]
labs = labs.select('encounter_id', 'lab_result_time', 'lab_name', *select_expr)

labs = labs.withColumn('meas_hour', f.hour(f.col('lab_result_time')))
labs = labs.withColumn('meas_date', f.to_date(f.col('lab_result_time')))

## Get min and max for each time lab
group_cols = ["encounter_id",'meas_date', 'meas_hour']
labs_wide = labs.groupBy(group_cols) \
                  .pivot("lab_name") \
                  .agg(f.min('lab_value').alias("min"),
                  f.max('lab_value').alias("max")).orderBy(group_cols)

## Join to cohort hourly blocked dataset
cohort_hours = spark.read.parquet("/project2/wparker/SIPA_data/life_support_cohort_48.parquet")

group_cols = ["encounter_id", 'meas_date', 'meas_hour']
cohort_labs_48 = cohort_hours.join(labs_wide, on=group_cols, how='left')


cohort_labs_48.write.parquet("/project2/wparker/SIPA_data/cohort_labs_48.parquet", mode="overwrite")

cohort_labs.repartition(1).write.parquet("/project2/wparker/SIPA_data/cohort_labs_48.parquet", mode="overwrite")

