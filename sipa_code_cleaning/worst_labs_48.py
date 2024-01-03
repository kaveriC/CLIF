from pyspark.sql import SparkSession
import pyspark.sql.functions as f
from pyspark import SparkConf
import gc
from pyspark.sql.window import Window



print("loaded libraries")
spark = SparkSession.builder \
        .appName("worst labs") \
        .getOrCreate()

## Read in Labs
labs = spark.read.parquet("/project2/wparker/SIPA_data/RCLIF_labs_10312023.parquet")
labs = labs.withColumn('lab_result_time',f.to_timestamp('lab_order_time','yyyy-MM-dd HH:mm:ss'))
labs = labs.select('C19_HAR_ID', 'lab_result_time','lab_name', 'lab_value')

### Cleaning up values/columns
select_expr = [f.regexp_replace(f.col('lab_name'), "[\ufeff]", "").alias('lab_name')]
labs = labs.select('C19_HAR_ID', 'lab_result_time', 'lab_value', *select_expr)

labs = labs.withColumn('lab_result_time',f.to_timestamp('lab_result_time','yyyy-MM-dd HH:mm:ss'))

select_expr = [f.regexp_replace(f.col('lab_value'), "[\ufeff]", "").alias('lab_value')]
labs = labs.select('C19_HAR_ID', 'lab_result_time', 'lab_name', *select_expr)

select_expr = [f.regexp_replace(f.col('lab_value'), "[<]", "").alias('lab_value')]
labs = labs.select('C19_HAR_ID', 'lab_result_time', 'lab_name', *select_expr)

select_expr = [f.regexp_replace(f.col('lab_value'), "[>]", "").alias('lab_value')]
labs = labs.select('C19_HAR_ID', 'lab_result_time', 'lab_name', *select_expr)

labs = labs.withColumn('meas_hour', f.hour(f.col('lab_result_time')))
labs = labs.withColumn('meas_date', f.to_date(f.col('lab_result_time')))
labs = labs.withColumn("lab_value_num",labs.lab_value.cast('double'))

## Get first and last measurement times per person
labs_hours = labs.select("C19_HAR_ID","lab_result_time").distinct()
labs_hours = labs_hours.groupBy('C19_HAR_ID').agg((f.min('lab_result_time').alias("observed_labs_start")),
                                               (f.max('lab_result_time').alias("observed_labs_end")))

labs_hours = labs_hours.withColumn('observed_labs_start',f.to_timestamp('observed_labs_start','yyyy-MM-dd'))
labs_hours = labs_hours.withColumn('observed_labs_end',f.to_timestamp('observed_labs_end','yyyy-MM-dd'))

## Explode between first and last measurement times to get all hourly timestamps
labs_hours = labs_hours.withColumn('txnDt', 
                                   f.explode(f.expr('sequence(observed_labs_start, observed_labs_end, interval 1 hour)')))
labs_hours = labs_hours.withColumn('meas_hour', f.hour(f.col('txnDt')))
labs_hours = labs_hours.withColumn('meas_date', f.to_date(f.col('txnDt')))
labs_hours = labs_hours.select('C19_HAR_ID', 'txnDt', 'meas_date', 'meas_hour', 
                               'observed_labs_start', 'observed_labs_end')

labs = labs.select('C19_HAR_ID', 'meas_date', 'meas_hour', 'lab_name', 'lab_value_num')

## Read in cohort
cohort = spark.read.parquet("/project2/wparker/SIPA_data/life_support_cohort.parquet")
cohort = cohort.withColumn('life_support_start_time',f.to_timestamp('life_support_start','yyyy-MM-dd HH:mm:ss'))
cohort = cohort.select('C19_HAR_ID', 'life_support_start_time')

group_cols = ["C19_HAR_ID"]
cohort_labs = cohort.join(labs, on=group_cols, how="left")

cohort_labs = cohort_labs.select('C19_HAR_ID', 'life_support_start_time','meas_date', 'meas_hour', 
                                       'lab_name', 'lab_value_num')
cohort_labs = cohort_labs.filter(f.col('lab_name').isNotNull())

## Get min and max for each time lab
group_cols = ["C19_HAR_ID", "life_support_start_time",'meas_date', 'meas_hour']
cohort_labs_wide = cohort_labs.groupBy(group_cols) \
                                     .pivot("lab_name") \
                                     .agg(f.min('lab_value_num').alias("min"),
                                         f.max('lab_value_num').alias("max")).orderBy(group_cols)

cohort_hours = cohort.join(labs_hours, on='C19_HAR_ID', how='left')



group_cols = ["C19_HAR_ID", 'life_support_start_time', 'meas_date', 'meas_hour']
cohort_labs_wide = cohort_hours.join(cohort_labs_wide, on=group_cols, how='full').orderBy(group_cols)

## Carry forward labs we need for SIPA

cohort_labs_wide_2 = cohort_labs_wide.withColumn('billirubin_max_filled', 
                                       f.coalesce(f.col('bilirubin_total_max'), 
                                                  f.last('bilirubin_total_max', True) \
                                                  .over(Window.partitionBy('C19_HAR_ID') \
                                                        .orderBy('txnDt')), f.lit('NULL')))


cohort_labs_wide_2 = cohort_labs_wide_2.withColumn('platelet_count_min_filled', 
                                       f.coalesce(f.col('platelet_count_min'), 
                                                  f.last('platelet_count_min', True) \
                                                  .over(Window.partitionBy('C19_HAR_ID') \
                                                        .orderBy('txnDt')), f.lit('NULL')))


cohort_labs_wide_2 = cohort_labs_wide_2.withColumn('creatinine_max_filled', 
                                       f.coalesce(f.col('creatinine_max'), 
                                                  f.last('creatinine_max', True) \
                                                  .over(Window.partitionBy('C19_HAR_ID') \
                                                        .orderBy('txnDt')), f.lit('NULL')))

cohort = cohort_labs_wide_2.withColumn('window_start', (f.col('life_support_start_time')-f.expr("INTERVAL 41 HOURS")))
cohort = cohort.withColumn('window_end', (f.col('life_support_start_time')+f.expr("INTERVAL 6 HOURS")))

cohort_labs_48 = cohort.filter((f.col('txnDt') >= f.col('window_start')) &
                              (f.col('txnDt') <= f.col('window_end')))

cohort_labs_48.write.parquet("/project2/wparker/SIPA_data/cohort_labs_48.parquet", mode="overwrite")

group_cols = ['C19_HAR_ID', 'life_support_start_time', 'window_start', 'window_end',
              'observed_labs_start', 'observed_labs_end']

cohort_labs_48_summary = cohort_labs_48.groupBy(group_cols) \
                                     .agg(f.min('creatinine_max_filled').alias("creatinine_max_filled"),
                                         f.max('platelet_count_min_filled').alias("platelet_count_min_filled"),
                                         f.max('billirubin_max_filled').alias("billirubin_max_filled"))\
                                    .distinct()\
                                    .orderBy("life_support_start_time")

cohort_labs_48_summary.write.parquet("/project2/wparker/SIPA_data/cohort_labs_48_summary.parquet", 
                                       mode="overwrite")
