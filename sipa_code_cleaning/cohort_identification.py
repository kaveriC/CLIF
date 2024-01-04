#######  Set up spark session ##############
import pyarrow as pa
import pyarrow.parquet as pq
from pyspark.sql import SparkSession
import pyspark.sql.functions as f
from pyspark import SparkConf
from pyspark.sql.window import Window



print("loaded libraries")
spark = SparkSession.builder \
        .appName("cohort identification") \
        .getOrCreate()
#############################################

######## Load in encounter dataset
demo_disp = spark.read.parquet("/project2/wparker/SIPA_data/RCLIF_patient_enc_demo_dispo.parquet")
demo_disp = demo_disp.withColumn('adm_date',f.to_date('adm_date','yyyy-MM-dd'))

# Filter to time period, adults only

demo_disp = demo_disp.filter(((f.col('adm_date')>='2020-03-01') & 
                   (f.col('adm_date')<='2022-03-31')))
demo_disp = demo_disp.filter(f.col('age_at_adm')>=18)

######### Get worst FiO2

## Read in respiratory support table
resp_full = spark.read.option("header",True).csv('/project2/wparker/SIPA_data/RCLIF_respiratory_support_09282023.csv')
resp_full = resp_full.withColumn('recorded_time',f.to_timestamp('recorded_time','yyyy-MM-dd HH:mm:ss'))

resp_full = resp_full.withColumn("fio2",resp_full.fio2.cast('double'))
resp_full = resp_full.withColumn("lpm",resp_full.lpm.cast('double'))


## Filter for valid values or Null
resp_full = resp_full.filter((((f.col('fio2')>=0.21) &
                              (f.col('fio2')<=1)) |
                              (f.col('fio2').isNull())))

## Filter out people on NIPPV without a FiO2 measurement--CPAP
resp_full = resp_full.filter(~((f.col('device_name')=='NIPPV') &
                              (f.col('fio2').isNull())))

## Replace NA/Null strings with actual Nulls
resp_full = resp_full.withColumn('device_name', f.when(~f.col('device_name').rlike(r'NA'), f.col('device_name')))
resp_full = resp_full.withColumn('device_name', f.when(~f.col('device_name').rlike(r'NULL'), f.col('device_name')))

resp_full = resp_full.withColumn('device_name_2', f.expr(
        """
        CASE
        WHEN device_name IS NOT NULL THEN device_name
        WHEN device_name IS NULL AND fio2 ==.21 AND lpm IS NULL and peep=='NA' THEN 'Room Air'
        WHEN device_name IS NULL AND fio2 IS NULL AND lpm IS NULL THEN 'Room Air'
        WHEN device_name IS NULL AND fio2 IS NULL AND lpm ==0 THEN 'Room Air'
        WHEN device_name IS NULL AND fio2 IS NULL and lpm <=20 THEN 'Nasal Cannula'
        WHEN device_name IS NULL AND fio2 IS NULL and lpm >20 and peep=='NA' THEN 'High Flow NC'
        ELSE NULL
        END
        """
))

resp_full = resp_full.withColumn('fio2_combined', f.expr(
        """
        CASE
        WHEN fio2 IS NOT NULL THEN fio2
        WHEN fio2 IS NULL AND device_name_2 == 'Room Air' THEN .21
        WHEN fio2 IS NULL AND device_name_2 == 'Nasal Cannula' THEN ( 0.24 + (0.04 * lpm) )
        ELSE NULL
        END
        """
))

# Carry forward device & FiO2

## Get first and last measurement times per person
fio2_hours = fio2.select("C19_HAR_ID","meas_date").distinct()
fio2_hours = fio2_hours.groupBy('C19_HAR_ID').agg((f.min('meas_date').alias("first_date")),
                                               (f.max('meas_date').alias("last_date")))

fio2_hours = fio2_hours.withColumn('first_date',f.to_timestamp('first_date','yyyy-MM-dd'))
fio2_hours = fio2_hours.withColumn('last_date',f.to_timestamp('last_date','yyyy-MM-dd'))

## Explode between first and last measurement times to get all hourly timestamps
fio2_hours = fio2_hours.withColumn('txnDt', f.explode(f.expr('sequence(first_date, last_date, interval 1 hour)')))
fio2_hours = fio2_hours.withColumn('meas_hour', f.hour(f.col('txnDt')))
fio2_hours = fio2_hours.withColumn('meas_date', f.to_date(f.col('txnDt')))
fio2_hours = fio2_hours.select('C19_HAR_ID', 'txnDt', 'meas_date', 'meas_hour')

## Join to cohort max FiO2 
group_cols = ["C19_HAR_ID", "meas_date", "meas_hour"]
fio2_hours = fio2_hours.join(fio2, on=group_cols, how='left').orderBy('C19_HAR_ID', 'txnDt')

## Extract hour and date for blocking
resp_full = resp_full.select('C19_HAR_ID', 'device_name_2', 'recorded_time', 'fio2_combined', 'lpm')
resp_full = resp_full.withColumn('meas_hour', f.hour(f.col('recorded_time')))
resp_full = resp_full.withColumn('meas_date', f.to_date(f.col('recorded_time')))

fio2 = resp_full.select('C19_HAR_ID', 'device_name_2', 'meas_date', 'meas_hour', 'fio2_combined', 'lpm')

## Group by person, device, measurement date and measurement hour; get max FiO2 and LPM within each hour
group_cols = ["C19_HAR_ID", "device_name_2", "meas_date", "meas_hour"]
fio2 = fio2.groupBy(group_cols) \
            .agg((f.max('fio2_combined').alias("fio2_combined")),
                  (f.max('lpm').alias("lpm")))

## Left join back to only the encounters in the time frame
fio2 = fio2.join(demo_disp, on='C19_HAR_ID', how='leftsemi')

## Carry forward device name until another device is recorded or the end of the measurement time window
fio2_hours_2 = fio2_hours.withColumn('device_filled', 
                                       f.coalesce(f.col('device_name_2'), 
                                                  f.last('device_name_2', True) \
                                                  .over(Window.partitionBy('C19_HAR_ID') \
                                                        .orderBy('txnDt')), f.lit('NULL')))
fio2_hours_2 = fio2_hours_2.withColumn('device_filled', 
                                       f.when(~f.col('device_filled').rlike(r'NULL'), f.col('device_filled')))

## Carry forward FiO2 measurement name until another device is recorded or the end of the measurement time window
fio2_hours_2 = fio2_hours_2.withColumn("fio2_combined",fio2_hours_2.fio2_combined.cast('double'))

fio2_hours_2 = fio2_hours_2.withColumn('fio2_filled', 
                                       f.when((f.col('fio2_combined').isNotNull()), f.col('fio2_combined')))
fio2_hours_2 = fio2_hours_2.withColumn('fio2_filled', 
                                       f.coalesce(f.col('fio2_combined'), 
                                                  f.last('fio2_combined', True) \
                                                  .over(Window.partitionBy('C19_HAR_ID', 'device_filled') \
                                                        .orderBy('txnDt')), f.lit('NULL')))

fio2_filled = fio2_hours_2.select('C19_HAR_ID','txnDt','meas_date', 'meas_hour',
                                  'device_filled','fio2_filled')

# Now need PaO2
labs = spark.read.parquet("/project2/wparker/SIPA_data/RCLIF_labs_10312023.parquet")


### Cleaning up values/columns
labs = labs.select('C19_HAR_ID', 'lab_result_time','lab_name', 'lab_value')

select_expr = [f.regexp_replace(f.col('lab_name'), "[\ufeff]", "").alias('lab_name')]
labs = labs.select('C19_HAR_ID', 'lab_result_time', 'lab_value', *select_expr)

labs = labs.filter(f.col("lab_name")=="pao2")

labs = labs.withColumn('lab_result_time',f.to_timestamp('lab_result_time','yyyy-MM-dd HH:mm:ss'))

select_expr = [f.regexp_replace(f.col('lab_value'), "[\ufeff]", "").alias('lab_value')]
labs = labs.select('C19_HAR_ID', 'lab_result_time', 'lab_name', *select_expr)

select_expr = [f.regexp_replace(f.col('lab_value'), "[<]", "").alias('lab_value')]
labs = labs.select('C19_HAR_ID', 'lab_result_time', 'lab_name', *select_expr)

select_expr = [f.regexp_replace(f.col('lab_value'), "[>]", "").alias('lab_value')]
labs = labs.select('C19_HAR_ID', 'lab_result_time', 'lab_name', *select_expr)

labs = labs.withColumn('meas_hour', f.hour(f.col('lab_result_time')))
labs = labs.withColumn('meas_date', f.to_date(f.col('lab_result_time')))
labs = labs.select('C19_HAR_ID', 'meas_date', 'meas_hour', 'lab_name', 'lab_value')
labs = labs.withColumn("lab_value_num",labs.lab_value.cast('double'))


# Get min PaO2 per hour
group_cols = ["C19_HAR_ID","meas_date", "meas_hour"]
labs = labs.groupBy(group_cols) \
           .pivot("lab_name") \
           .agg(f.min('lab_value_num').alias("min"))
labs = labs.join(demo_disp, on='C19_HAR_ID', how='leftsemi')

## Get first and last measurement times per person
labs_hours = labs.select("C19_HAR_ID","meas_date").distinct()
labs_hours = labs_hours.groupBy('C19_HAR_ID').agg((f.min('meas_date').alias("first_date")),
                                               (f.max('meas_date').alias("last_date")))

labs_hours = labs_hours.withColumn('first_date',f.to_timestamp('first_date','yyyy-MM-dd'))
labs_hours = labs_hours.withColumn('last_date',f.to_timestamp('last_date','yyyy-MM-dd'))

## Explode between first and last measurement times to get all hourly timestamps
labs_hours = labs_hours.withColumn('txnDt', f.explode(f.expr('sequence(first_date, last_date, interval 1 hour)')))
labs_hours = labs_hours.withColumn('meas_hour', f.hour(f.col('txnDt')))
labs_hours = labs_hours.withColumn('meas_date', f.to_date(f.col('txnDt')))
labs_hours = labs_hours.select('C19_HAR_ID', 'txnDt', 'meas_date', 'meas_hour')

labs_hours = labs_hours.join(labs, on=group_cols, how='left').orderBy('C19_HAR_ID', 'txnDt')

## Carry forward PaO2 until next measurement or end of window, maximum 4 hours

### Get time of most recent PaO2
labs_hours_2 = labs_hours.withColumn('last_measure', f.when(f.col('pao2').isNotNull(), f.col('txnDt')))
labs_hours_2 = labs_hours_2.withColumn('last_measure', f.coalesce(f.col('last_measure'), 
                                                                  f.last('last_measure', True)\
                                                                  .over(Window.partitionBy('C19_HAR_ID')\
                                                                        .orderBy('txnDt')), f.lit('NULL')))

labs_hours_2 = labs_hours_2.withColumn('last_measure',f.to_timestamp('last_measure','yyyy-MM-dd HH:mm:ss'))
labs_hours_2 = labs_hours_2.withColumn('txnDt',f.to_timestamp('txnDt','yyyy-MM-dd HH:mm:ss'))

### Calculate time difference between the hour we're trying to fill and the most recent PaO2, filter to only 4 hrs
labs_hours_2 = labs_hours_2.withColumn("hour_diff", 
                                       (f.col("txnDt").cast("long")-f.col("last_measure").cast("long"))/(60*60))
labs_hours_2 = labs_hours_2.filter((f.col('hour_diff')>=0)&(f.col('hour_diff')<=3))

labs_hours_2 = labs_hours_2.withColumn("pao2_num",labs_hours_2.pao2.cast('double'))
labs_hours_2 = labs_hours_2.filter(f.col('pao2_num')>0)

### Fill PaO2 forward
labs_hours_2 = labs_hours_2.withColumn('pao2_filled', f.when(f.col('pao2_num').isNotNull(), f.col('pao2_num')))
labs_hours_2 = labs_hours_2.withColumn('pao2_filled', f.coalesce(f.col('pao2_num'), 
                                                                 f.last('pao2_num', True)\
                                                                 .over(Window.partitionBy('C19_HAR_ID', 
                                                                                          'last_measure')\
                                                                       .orderBy('txnDt')), f.lit('NULL')))

pao2_filled = labs_hours_2.select('C19_HAR_ID','txnDt','meas_date', 'meas_hour', 'pao2_filled')

# Now need spO2
vitals = spark.read.parquet("/project2/wparker/SIPA_data/RCLIF_vitals_10242023.parquet")
vitals = vitals.withColumn('measured_time',f.to_timestamp('recorded_time','yyyy-MM-dd HH:mm:ss'))
vitals = vitals.select('C19_HAR_ID', 'measured_time','vital_name', 'vital_value')

vitals = vitals.filter(f.col("vital_name")=="spO2")

vitals = vitals.withColumn('meas_hour', f.hour(f.col('measured_time')))
vitals = vitals.withColumn('meas_date', f.to_date(f.col('measured_time')))
vitals = vitals.select('C19_HAR_ID', 'meas_date', 'meas_hour', 'vital_name', 'vital_value')

# Get min SpO2 per hour

group_cols = ["C19_HAR_ID","meas_date", "meas_hour"]
vitals = vitals.groupBy(group_cols) \
           .pivot("vital_name") \
           .agg(f.min('vital_value').alias("min"))
vitals = vitals.join(demo_disp, on='C19_HAR_ID', how='leftsemi')

## Get first and last measurement times per person
vitals_hours = vitals.select("C19_HAR_ID","meas_date").distinct()
vitals_hours = vitals_hours.groupBy('C19_HAR_ID').agg((f.min('meas_date').alias("first_date")),
                                               (f.max('meas_date').alias("last_date")))

vitals_hours = vitals_hours.withColumn('first_date',f.to_timestamp('first_date','yyyy-MM-dd'))
vitals_hours = vitals_hours.withColumn('last_date',f.to_timestamp('last_date','yyyy-MM-dd'))

## Explode between first and last measurement times to get all hourly timestamps
vitals_hours = vitals_hours.withColumn('txnDt', f.explode(f.expr('sequence(first_date, last_date, interval 1 hour)')))
vitals_hours = vitals_hours.withColumn('meas_hour', f.hour(f.col('txnDt')))
vitals_hours = vitals_hours.withColumn('meas_date', f.to_date(f.col('txnDt')))
vitals_hours = vitals_hours.select('C19_HAR_ID', 'txnDt', 'meas_date', 'meas_hour')

vitals_hours = vitals_hours.join(vitals, on=group_cols, how='left').orderBy('C19_HAR_ID', 'txnDt')


vitals_hours_2 = vitals_hours.withColumn('last_measure', f.when(f.col('spO2').isNotNull(), f.col('txnDt')))
vitals_hours_2 = vitals_hours_2.withColumn('last_measure', 
                                           f.coalesce(f.col('last_measure'), 
                                                      f.last('last_measure', True)\
                                                      .over(Window.partitionBy('C19_HAR_ID')\
                                                            .orderBy('txnDt')), f.lit('NULL')))

vitals_hours_2 = vitals_hours_2.withColumn('last_measure',f.to_timestamp('last_measure','yyyy-MM-dd HH:mm:ss'))
vitals_hours_2 = vitals_hours_2.withColumn('txnDt',f.to_timestamp('txnDt','yyyy-MM-dd HH:mm:ss'))

## Get only valid values before filling
vitals_hours_2 = vitals_hours_2.withColumn("spO2_num",vitals_hours_2.spO2.cast('double'))
vitals_hours_2 = vitals_hours_2.filter(f.col('spO2_num')>60)
vitals_hours_2 = vitals_hours_2.filter(f.col('spO2_num')<=100)

## Cary forward SpO2
vitals_hours_2 = vitals_hours_2.withColumn('spO2_filled', 
                                           f.when(f.col('spO2_num').isNotNull(), f.col('spO2_num')))
vitals_hours_2 = vitals_hours_2.withColumn('spO2_filled', 
                                           f.coalesce(f.col('spO2_num'), 
                                                      f.last('spO2_num', True)\
                                                      .over(Window.partitionBy('C19_HAR_ID', 'last_measure')\
                                                            .orderBy('txnDt')), f.lit('NULL')))

spO2_filled = vitals_hours_2.select('C19_HAR_ID','txnDt','meas_date', 'meas_hour', 'spO2_filled')

# Merge FiO2, PaO2, spO2 to get FiO2/PaO2

fio2_filled = fio2_filled.repartition('C19_HAR_ID')
pao2_filled = pao2_filled.repartition('C19_HAR_ID')
spO2_filled = spO2_filled.repartition('C19_HAR_ID')

group_cols = ["C19_HAR_ID","txnDt","meas_date", "meas_hour"]
df = fio2_filled.join(pao2_filled, on=group_cols, how='full')
df = df.join(spO2_filled, on=group_cols, how='full')

df = df.withColumn("fio2_filled",df.fio2_filled.cast('double'))
df = df.withColumn("pao2_filled",df.pao2_filled.cast('double'))
df = df.withColumn("spO2_filled",df.spO2_filled.cast('double'))

# Get first time on oxygen support & P/F <200
df = df.withColumn("p_f", f.expr(
        """
        CASE
        WHEN fio2_filled IS NOT NULL AND pao2_filled IS NOT NULL THEN ( pao2_filled / fio2_filled )
        ELSE NULL
        END
        """
    ))

df = df.withColumn("s_f", f.expr(
        """
        CASE
        WHEN fio2_filled IS NOT NULL AND spO2_filled IS NOT NULL THEN ( spO2_filled / fio2_filled )
        ELSE NULL
        END
        """
    ))

df = df.distinct()
df.write.parquet("/project2/wparker/SIPA_data/p_f_combined_filled.parquet", mode="overwrite")

## Get first time somone on oxygen therapy with PaO2/FiO2 < 200 (S/F < 179 if no P/F measured)

df = df.filter((((f.col("p_f")<200))|
                (f.col("s_f")<179)))
df = df.filter(f.col("device_filled")!="NULL")
df = df.filter(f.col("device_filled")!="Room Air")
df = df.filter(f.col("device_filled")!="Vent")
df = df.filter(f.col("device_filled")!="NIPPV")
df = df.filter(f.col("device_filled").isNotNull())


df = df.select("C19_HAR_ID", "txnDt", "meas_date", "meas_hour", "device_filled","pao2_filled","fio2_filled",
              "spO2_filled")

w1 = Window.partitionBy("C19_HAR_ID").orderBy('txnDt')

df_first_with_time = df.withColumn("row",f.row_number().over(w1)) \
             .filter(f.col("row") == 1).drop("row")

df_first_with_time = df_first_with_time.select("C19_HAR_ID", "txnDt").withColumnRenamed("txnDt", "recorded_time")

#get just invasive or non-invasive mechanical ventilation
vent = resp_full.filter(((f.col('device_name')=='Vent') | 
                   (f.col('device_name')=='NIPPV')))

# minimum time by person

w3 = Window.partitionBy("C19_HAR_ID").orderBy('recorded_time')

vent_first = vent.withColumn("row",f.row_number().over(w3)) \
             .filter(f.col("row") == 1).drop("row")

# Merge with oxygen support and P/F < 200 group, get first time meeting criteria
vent_first = vent_first.repartition('C19_HAR_ID')
df_first_with_time = df_first_with_time.repartition('C19_HAR_ID')

group_cols = ["C19_HAR_ID","recorded_time"]
df = vent_first.join(df_first_with_time, on=group_cols, how='full')

resp_support = df.groupBy("C19_HAR_ID").agg(f.min("recorded_time").alias("resp_life_support_start")).distinct()

# Now pressors
df_meds = spark.read.option("header",True).csv('/project2/wparker/SIPA_data/RCLIF_meds_admin_conti.csv')
df_meds = df_meds.withColumn('admin_time',f.to_timestamp('admin_time','yyyy-MM-dd HH:mm:ss'))

# Filter to pressor medications

pressors = df_meds.filter(((f.col('med_name')=='phenylephrine') | 
                       (f.col('med_name')=='epinephrine') | 
                       (f.col('med_name')=='vasopressin') | 
                       (f.col('med_name')=='dopamine') |
                       (f.col('med_name')=='dobutamine') |
                       (f.col('med_name')=='norepinephrine') |
                       (f.col('med_name')=='angiotensin') |
                       (f.col('med_name')=='isoproterenol')|
                        (f.col('med_name')=='milrinone')))
pressors = pressors.select("C19_HAR_ID", "admin_time")

# Get first time someone is on a pressor
pressors = pressors.groupBy("C19_HAR_ID").agg(f.min("admin_time").alias("pressor_life_support_start"))
pressors = pressors.join(demo_disp, on='C19_HAR_ID', how='leftsemi').distinct()

df = pressors.join(resp_support, on='C19_HAR_ID', how='full')
df = df.withColumn("life_support_start", f.least(f.col('pressor_life_support_start'),
                                                 f.col('resp_life_support_start')))
df = df.select('C19_HAR_ID', 'life_support_start')
df = df.join(demo_disp, on='C19_HAR_ID', how='inner').orderBy('adm_date').distinct()
df = df.filter(f.col('life_support_start').isNotNull())

df.write.parquet("/project2/wparker/SIPA_data/life_support_cohort.parquet", mode="overwrite")

