from pyspark.sql import SparkSession
import pyspark.sql.functions as f
from pyspark.sql.window import Window


print("loaded libraries")
spark = SparkSession.builder \
        .appName("worst vitals") \
        .getOrCreate()

vitals = spark.read.parquet("/project2/wparker/SIPA_data/RCLIF_vitals_10242023.parquet")
vitals = vitals.withColumn('measured_time',f.to_timestamp('recorded_time','yyyy-MM-dd HH:mm:ss'))

vitals = vitals.withColumn('meas_hour', f.hour(f.col('measured_time')))
vitals = vitals.withColumn('meas_date', f.to_date(f.col('measured_time')))
vitals = vitals.withColumn("vital_value_num",vitals.vital_value.cast('double'))

## Get first and last measurement times per person
vitals_hours = vitals.select("C19_HAR_ID","measured_time").distinct()
vitals_hours = vitals_hours.groupBy('C19_HAR_ID').agg((f.min('measured_time').alias("observed_vitals_start")),
                                               (f.max('measured_time').alias("observed_vitals_end")))

vitals_hours = vitals_hours.withColumn('observed_vitals_start',f.to_timestamp('observed_vitals_start','yyyy-MM-dd'))
vitals_hours = vitals_hours.withColumn('observed_vitals_end',f.to_timestamp('observed_vitals_end','yyyy-MM-dd'))

## Explode between first and last measurement times to get all hourly timestamps
vitals_hours = vitals_hours.withColumn('txnDt', 
                                   f.explode(f.expr('sequence(observed_vitals_start, observed_vitals_end, interval 1 hour)')))
vitals_hours = vitals_hours.withColumn('meas_hour', f.hour(f.col('txnDt')))
vitals_hours = vitals_hours.withColumn('meas_date', f.to_date(f.col('txnDt')))
vitals_hours = vitals_hours.select('C19_HAR_ID', 'txnDt', 'meas_date', 'meas_hour', 
                               'observed_vitals_start', 'observed_vitals_end')

vitals = vitals.select('C19_HAR_ID', 'meas_date', 'meas_hour', 'vital_name', 'vital_value_num')
vitals = vitals.filter(f.col('vital_value_num')>=0)

## Read in cohort
cohort = spark.read.parquet("/project2/wparker/SIPA_data/life_support_cohort.parquet")
cohort = cohort.withColumn('life_support_start_time',f.to_timestamp('life_support_start','yyyy-MM-dd HH:mm:ss'))
cohort = cohort.select('C19_HAR_ID', 'life_support_start_time')

group_cols = ["C19_HAR_ID"]
cohort_vitals = cohort.join(vitals, on=group_cols, how="left")

cohort_vitals = cohort_vitals.select('C19_HAR_ID', 'life_support_start_time','meas_date', 'meas_hour', 
                                       'vital_name', 'vital_value_num')
cohort_vitals = cohort_vitals.filter(f.col('vital_name').isNotNull())

## Get min and max for each hour
group_cols = ["C19_HAR_ID", "life_support_start_time",'meas_date', 'meas_hour']
cohort_vitals_wide = cohort_vitals.groupBy(group_cols) \
                                     .pivot("vital_name") \
                                     .agg(f.min('vital_value_num').alias("min"),
                                         f.max('vital_value_num').alias("max")).orderBy(group_cols)

cohort_hours = cohort.join(vitals_hours, on='C19_HAR_ID', how='left')



group_cols = ["C19_HAR_ID", 'life_support_start_time', 'meas_date', 'meas_hour']
cohort_vitals_wide = cohort_hours.join(cohort_vitals_wide, on=group_cols, how='full').orderBy(group_cols)

## Carry forward height and weight only for SIPA
cohort_vitals_wide_2 = cohort_vitals_wide.withColumn('weight_filled', 
                                       f.coalesce(f.col('weight_max'), 
                                                  f.last('weight_max', True) \
                                                  .over(Window.partitionBy('C19_HAR_ID') \
                                                        .orderBy('txnDt')), f.lit('NULL')))

cohort_vitals_wide_2 = cohort_vitals_wide_2.withColumn('height_filled', 
                                       f.coalesce(f.col('height_max'), 
                                                  f.last('height_max', True) \
                                                  .over(Window.partitionBy('C19_HAR_ID') \
                                                        .orderBy('txnDt')), f.lit('NULL')))

cohort = cohort_vitals_wide_2.withColumn('window_start', (f.col('life_support_start_time')-f.expr("INTERVAL 41 HOURS")))
cohort = cohort.withColumn('window_end', (f.col('life_support_start_time')+f.expr("INTERVAL 6 HOURS")))

cohort_vitals_48 = cohort.filter((f.col('txnDt') >= f.col('window_start')) &
                              (f.col('txnDt') <= f.col('window_end')))
cohort_vitals_48 = cohort_vitals_48.withColumn("MAP_for_sofa", f.expr(
        """
        CASE
        WHEN MAP_min IS NOT NULL THEN MAP_min
        WHEN MAP_min IS NULL AND sbp_min IS NOT NULL AND dbp_min IS NOT NULL THEN ( sbp_min + 2.0 * dbp_min ) / 3.0
        ELSE NULL
        END
        """
    ))
cohort_vitals_48.write.parquet("/project2/wparker/SIPA_data/cohort_vitals_48.parquet", mode="overwrite")

group_cols = ['C19_HAR_ID', 'life_support_start_time', 'window_start', 'window_end',
              'observed_vitals_start', 'observed_vitals_end']

cohort_vitals_48_summary = cohort_vitals_48.groupBy(group_cols) \
                                     .agg(f.max('weight_filled').alias("weight_filled"),
                                         f.max('height_filled').alias("height_filled"),
                                         f.min('MAP_for_sofa').alias("MAP_for_sofa"))\
                                    .distinct()\
                                    .orderBy("life_support_start_time")

cohort_vitals_48_summary.write.parquet("/project2/wparker/SIPA_data/cohort_vitals_48_summary.parquet", 
                                         mode="overwrite")




