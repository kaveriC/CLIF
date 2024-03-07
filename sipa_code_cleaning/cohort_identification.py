#######  Set up spark session ##############
import os
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
        .appName("cohort identification") \
        .config("spark.driver.memory", "15G") \
        .getOrCreate()
#############################################
######## Set directory ######## 
#############################################
path = "/project2/wparker/SIPA_data/"
os.chdir(r'/project2/wparker/SIPA_data/')
print(os.chdir)


#############################################
######## Identify Cohort ########
#############################################
# Load in limited IDs for admission dates in time period
limited_ids = spark.read.parquet("RCLIF_limited_identifers.parquet")
limited_ids = limited_ids.select('encounter_id', 'admission_dttm', 'discharge_dttm', 'zip_code').distinct()

# Filter to time period
limited_ids = limited_ids.filter(((f.col('admission_dttm')>='2020-03-01') & 
                   (f.col('admission_dttm')<='2022-03-31')))
limited_ids = limited_ids.filter((f.col('discharge_dttm')>=f.col('admission_dttm')))
limited_ids = limited_ids.filter(f.col('discharge_dttm').isNotNull())
limited_ids = limited_ids.filter(f.col('admission_dttm').isNotNull())


# Load in encounters for age and discharge disposition
demo_disp = spark.read.parquet("RCLIF_encounter_demographics_dispo.parquet")
demo_disp = demo_disp.select('patient_id', 'encounter_id', 'age_at_admission', 'disposition')

# Filter to adults only
demo_disp = demo_disp.filter(f.col('age_at_admission')>=18)

adults_in_time = limited_ids.join(demo_disp, on='encounter_id', how='inner')

# Exclude people only in ER/OR
adt = spark.read.parquet("RCLIF_adt.parquet")

adult_ids = adults_in_time.select('encounter_id')
adt_adults = adult_ids.join(adt, on='encounter_id', how='inner')
adt_adults = adt_adults.select('encounter_id', 'location_name').distinct()
adt_adults = adt_adults.withColumn('count', f.lit(1))
adt_adults_wide = adt_adults.groupBy('encounter_id').pivot('location_name').agg(f.sum('count'))
adt_adults_wide = adt_adults_wide.filter(f.col('ICU').isNotNull() |
                                         f.col('null').isNotNull() |
                                         f.col('Ward').isNotNull())
adt_adults_wide = adt_adults_wide.select('encounter_id').distinct()

adults_in_time = adt_adults_wide.join(adults_in_time, on='encounter_id', how='left')
har_ids = adults_in_time.select('encounter_id', 'admission_dttm', 'discharge_dttm')

## Explode between admission and discharge to get all hourly timestamps
har_ids_hours = har_ids.withColumn('txnDt', f.explode(f.expr('sequence(admission_dttm, discharge_dttm, interval 1 hour)')))
har_ids_hours = har_ids_hours.withColumn('meas_hour', f.hour(f.col('txnDt')))
har_ids_hours = har_ids_hours.withColumn('meas_date', f.to_date(f.col('txnDt')))
har_ids_hours = har_ids_hours.select('encounter_id', 'meas_date', 'meas_hour').orderBy('encounter_id', 'meas_date', 'meas_hour')

######### Get worst FiO2
# Read in respiratory support table
resp_full = spark.read.parquet('RCLIF_resp_support.parquet')

# Filter to only adults in time frame
resp_full = har_ids.join(resp_full, on='encounter_id', how='inner')

# Filter to only variables we need
resp_full = resp_full.select('encounter_id', 'recorded_time', 'device_name','lpm', 'fio2', 'peep', 'set_volume')

# Filter for valid values or Null
resp_full = resp_full.filter((((f.col('fio2')>=0.21) &
                              (f.col('fio2')<=1)) |
                              (f.col('fio2').isNull())))

resp_full = resp_full.filter((((f.col('peep')>=3) &
                              (f.col('peep')<=30)) |
                              (f.col('peep').isNull())))

# Filter out people on CPAP with an FiO2 < 0.3
resp_full = resp_full.filter(~((f.col('device_name')=="CPAP") &
                             ((f.col('fio2').isNull()) | (f.col('fio2')<0.3))))


# Try to fill in some of the null device names based on other values
resp_full = resp_full.withColumn('device_name_2', f.expr(
        """
        CASE
        WHEN device_name IS NOT NULL THEN device_name
        WHEN device_name IS NULL AND fio2 ==.21 AND lpm IS NULL AND peep IS NULL AND set_volume IS NULL THEN 'Room Air'
        WHEN device_name IS NULL AND fio2 IS NULL AND lpm ==0 AND peep IS NULL AND set_volume IS NULL THEN 'Room Air'
        WHEN device_name IS NULL AND fio2 IS NULL AND lpm <=20 AND peep IS NULL AND set_volume IS NULL THEN 'Nasal Cannula'
        WHEN device_name IS NULL AND fio2 IS NULL AND lpm >20 AND peep IS NULL AND set_volume IS NULL THEN 'High Flow NC'
        WHEN device_name == "Nasal Cannula" AND fio2 IS NULL AND lpm >20 THEN 'High Flow NC'
        ELSE NULL
        END
        """
))

# Try to fill in FiO2 based on LPM for nasal cannula
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

# Filter for valid values or Null
resp_full = resp_full.filter((((f.col('fio2_combined')>=0.21) &
                              (f.col('fio2_combined')<=1)) |
                              (f.col('fio2_combined').isNull())))


### Carry forward device & FiO2

# Extract hour and date for blocking
resp_full = resp_full.select('encounter_id', 'device_name_2', 'recorded_time', 'fio2_combined', 'lpm')
resp_full = resp_full.withColumn('meas_hour', f.hour(f.col('recorded_time')))
resp_full = resp_full.withColumn('meas_date', f.to_date(f.col('recorded_time')))

fio2 = resp_full.select('encounter_id', 'device_name_2', 'meas_date', 'meas_hour', 'fio2_combined', 'lpm').distinct()

# Need to rank devices to get max in hour
fio2 = fio2.withColumn("device_rank", f.expr(
        """
        CASE
        WHEN device_name_2 == 'Vent' THEN 1
        WHEN device_name_2 == 'NIPPV' THEN 2
        WHEN device_name_2 == 'CPAP' THEN 3
        WHEN device_name_2 == 'High Flow NC' THEN 4
        WHEN device_name_2 == 'Face Mask' THEN 5
        WHEN device_name_2 == 'Trach Collar' THEN 6
        WHEN device_name_2 == 'Nasal Cannula' THEN 7
        WHEN device_name_2 == 'Other' THEN 8
        WHEN device_name_2 == 'Room Air' THEN 9
        WHEN device_name_2 IS NULL THEN NULL
        ELSE NULL
        END
        """
    ))

# Group by person, device, measurement date and measurement hour; get max FiO2 and LPM within each hour
group_cols = ["encounter_id", "meas_date", "meas_hour"]
fio2 = fio2.groupBy(group_cols) \
            .agg((f.max('fio2_combined').alias("fio2_combined")),
                  (f.min('device_rank').alias("device_rank")),
                  (f.max('lpm').alias("lpm"))).orderBy(group_cols)
fio2 = fio2.withColumn("device_name", f.expr(
        """
        CASE
        WHEN device_rank == 1 THEN 'Vent'
        WHEN device_rank == 2 THEN 'NIPPV' 
        WHEN device_rank == 3 THEN 'CPAP' 
        WHEN device_rank == 4 THEN 'High Flow NC'
        WHEN device_rank == 5 THEN 'Face Mask' 
        WHEN device_rank == 6 THEN 'Trach Collar'
        WHEN device_rank == 7 THEN 'Nasal Cannula'
        WHEN device_rank == 8 THEN 'Other'
        WHEN device_rank == 9 THEN 'Room Air'
        WHEN device_rank IS NULL THEN NULL
        ELSE NULL
        END
        """
    ))


# Merge back to hourly blocked cohort table 
group_cols = ["encounter_id", "meas_date", "meas_hour"]
fio2_hours = har_ids_hours.join(fio2, on=group_cols, how='left').orderBy(group_cols).distinct()


# Carry forward device name until another device is recorded or the end of the measurement time window
fio2_hours = fio2_hours.withColumn('device_filled', 
                                       f.coalesce(f.col('device_name'), 
                                                  f.last('device_name', True) \
                                                  .over(Window.partitionBy('encounter_id') \
                                                        .orderBy(group_cols)), f.lit('NULL')))
fio2_hours = fio2_hours.withColumn('device_filled', 
                                       f.when(~f.col('device_filled').rlike(r'NULL'), f.col('device_filled')))

# Carry forward FiO2 measurement name until another device is recorded or the end of the measurement time window
fio2_hours = fio2_hours.withColumn("fio2_combined",fio2_hours.fio2_combined.cast('double'))

fio2_hours = fio2_hours.withColumn('fio2_filled', 
                                       f.when((f.col('fio2_combined').isNotNull()), f.col('fio2_combined')))
fio2_hours = fio2_hours.withColumn('fio2_filled', 
                                       f.coalesce(f.col('fio2_combined'), 
                                                  f.last('fio2_combined', True) \
                                                  .over(Window.partitionBy('encounter_id', 'device_filled') \
                                                        .orderBy(group_cols)), f.lit('NULL')))

# Carry forward LPM measurement name until another device is recorded or the end of the measurement time window
fio2_hours = fio2_hours.withColumn("lpm",fio2_hours.lpm.cast('double'))

fio2_hours = fio2_hours.withColumn('lpm_filled', 
                                       f.when((f.col('lpm').isNotNull()), f.col('lpm')))
fio2_hours = fio2_hours.withColumn('lpm_filled', 
                                       f.coalesce(f.col('lpm'), 
                                                  f.last('lpm', True) \
                                                  .over(Window.partitionBy('encounter_id', 'device_filled') \
                                                        .orderBy(group_cols)), f.lit('NULL')))

fio2_filled = fio2_hours.select('encounter_id','meas_date', 'meas_hour',
                                  'device_filled','fio2_filled', 'lpm_filled').distinct()

######### Get worst PaO2
labs = spark.read.parquet("RCLIF_labs.parquet")

# Selecting only variables we need
labs = labs.select('encounter_id', 'lab_result_time','lab_name', 'lab_value')

# Cleaning up values
select_expr = [f.regexp_replace(f.col('lab_name'), "[\ufeff]", "").alias('lab_name')]
labs = labs.select('encounter_id', 'lab_result_time', 'lab_value', *select_expr)

select_expr = [f.regexp_replace(f.col('lab_value'), "[<]", "").alias('lab_value')]
labs = labs.select('encounter_id', 'lab_result_time', 'lab_name', *select_expr)

select_expr = [f.regexp_replace(f.col('lab_value'), "[>]", "").alias('lab_value')]
labs = labs.select('encounter_id', 'lab_result_time', 'lab_name', *select_expr)

labs = labs.withColumn('meas_hour', f.hour(f.col('lab_result_time')))
labs = labs.withColumn('meas_date', f.to_date(f.col('lab_result_time')))

# Filtering to only adults in time period
labs = labs.join(har_ids, on='encounter_id', how='inner')

# Filtering to only valid arterial PaO2, formatting values
pao2 = labs.filter(f.col("lab_name")=="pao2")
pao2 = pao2.withColumn("lab_value",pao2.lab_value.cast('double'))
pao2 = pao2.filter(f.col("lab_value")>30)
pao2 = pao2.withColumn('lab_result_time',f.to_timestamp('lab_result_time','yyyy-MM-dd HH:mm:ss'))

# Get min PaO2 per hour
pao2 = pao2.withColumn('meas_hour', f.hour(f.col('lab_result_time')))
pao2 = pao2.withColumn('meas_date', f.to_date(f.col('lab_result_time')))
pao2 = pao2.select('encounter_id', 'meas_date', 'meas_hour', 'lab_name', 'lab_value')

group_cols = ["encounter_id","meas_date", "meas_hour"]
pao2 = pao2.groupBy(group_cols) \
           .pivot("lab_name") \
           .agg(f.min('lab_value').alias("min"))

# Merge back to hourly blocked cohort table 
group_cols = ["encounter_id", "meas_date", "meas_hour"]
pao2_hours = har_ids_hours.join(pao2, on=group_cols, how='left').orderBy(group_cols).distinct()

pao2_hours = pao2_hours.withColumn("dttm", f.concat(pao2_hours.meas_date, f.lit(" "), pao2_hours.meas_hour))
pao2_hours = pao2_hours.withColumn('dttm',f.to_timestamp('dttm','yyyy-MM-dd HH'))

# Carry forward PaO2 until next measurement or end of window, maximum 4 hours

# Get time of most recent PaO2
pao2_hours = pao2_hours.withColumn('last_measure', f.when(f.col('pao2').isNotNull(), f.col('dttm')))
pao2_hours = pao2_hours.withColumn('last_measure', f.coalesce(f.col('last_measure'), 
                                                                  f.last('last_measure', True)\
                                                                  .over(Window.partitionBy('encounter_id')\
                                                                        .orderBy('dttm')), f.lit('NULL')))

pao2_hours = pao2_hours.withColumn('last_measure',f.to_timestamp('last_measure','yyyy-MM-dd HH:mm:ss'))

# Calculate time difference between the hour we're trying to fill and the most recent PaO2, filter to only 3 additional hrs (4 total)
pao2_hours = pao2_hours.withColumn("hour_diff", 
                                       (f.col("dttm").cast("long")-f.col("last_measure").cast("long"))/(60*60))
pao2_hours_2 = pao2_hours.filter((f.col('hour_diff')>=0)&(f.col('hour_diff')<=3))

# Fill PaO2 forward
pao2_hours_2 = pao2_hours_2.withColumn('pao2_filled', f.when(f.col('pao2').isNotNull(), f.col('pao2')))
pao2_hours_2 = pao2_hours_2.withColumn('pao2_filled', f.coalesce(f.col('pao2'), 
                                                                 f.last('pao2', True)\
                                                                 .over(Window.partitionBy('encounter_id', 
                                                                                          'last_measure')\
                                                                       .orderBy('dttm')), f.lit('NULL')))

pao2_filled = pao2_hours_2.select('encounter_id','meas_date', 'meas_hour', 'pao2_filled')

######### Get worst SpO2
vitals = spark.read.parquet("RCLIF_vitals.parquet")
vitals = vitals.filter(f.col('vital_name').isNotNull())
vitals = vitals.filter(f.col('vital_value').isNotNull())
vitals = vitals.filter(f.col('vital_value')>0)

# Filtering to only adults in time period
vitals = vitals.join(har_ids, on='encounter_id', how='inner')

# Selecting variables we need
vitals = vitals.select('encounter_id', 'recorded_time','vital_name', 'vital_value')
vitals = vitals.withColumn('meas_hour', f.hour(f.col('recorded_time')))
vitals = vitals.withColumn('meas_date', f.to_date(f.col('recorded_time')))

# Filtering to only SpO2, valid values
spo2 = vitals.filter(f.col("vital_name")=="spO2")
spo2 = spo2.select('encounter_id', 'meas_date', 'meas_hour', 'vital_name', 'vital_value')
spo2 = spo2.filter(f.col('vital_value')>=60)
spo2 = spo2.filter(f.col('vital_value')<=96)

# Get min SpO2 per hour
group_cols = ["encounter_id","meas_date", "meas_hour"]
spo2 = spo2.groupBy(group_cols) \
           .pivot("vital_name") \
           .agg(f.min('vital_value').alias("min"))

# Merge back to hourly blocked cohort table 
group_cols = ["encounter_id", "meas_date", "meas_hour"]
spo2_hours = har_ids_hours.join(spo2, on=group_cols, how='left').distinct()
spo2_hours = spo2_hours.select('encounter_id','meas_date', 'meas_hour', 'spO2')

# Merge FiO2, PaO2, spO2 to get FiO2/PaO2
fio2_filled = fio2_filled.repartition('encounter_id')
pao2_filled = pao2_filled.repartition('encounter_id')
spo2_hours = spo2_hours.repartition('encounter_id')

group_cols = ["encounter_id","meas_date", "meas_hour"]
df = fio2_filled.join(pao2_filled, on=group_cols, how='left')
df = df.join(spo2_hours, on=group_cols, how='left').orderBy(group_cols)

df = df.withColumn("fio2_filled",df.fio2_filled.cast('double'))
df = df.withColumn("pao2_filled",df.pao2_filled.cast('double'))
df = df.withColumn("lpm_filled",df.lpm_filled.cast('double'))

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
        WHEN fio2_filled IS NOT NULL AND spO2 IS NOT NULL THEN ( spO2 / fio2_filled )
        ELSE NULL
        END
        """
    ))

df = df.distinct()

############ Get number of pressors on per hour
df_meds = spark.read.parquet('RCLIF_medication_admin_continuous.parquet')
df_meds = df_meds.withColumn('meas_hour', f.hour(f.col('admin_dttm')))
df_meds = df_meds.withColumn('meas_date', f.to_date(f.col('admin_dttm')))

# Filter to pressor medications
pressors = df_meds.filter(f.col('med_category')=='vasoactives')
pressors = pressors.select("encounter_id", "meas_hour", "meas_date", "med_name").distinct()

# Filtering to only adults in time period
pressors = pressors.join(har_ids, on='encounter_id', how='inner')

# Get max num pressors (cap at 4) per hour
w2 = Window.partitionBy(group_cols).orderBy("med_name")

group_cols = ['encounter_id','meas_hour','meas_date', 'med_name']

cohort_pressors_grouped = pressors.withColumn("row",f.row_number().over(w2))
cohort_pressors_grouped = pressors.withColumn("count",f.lit(1))

group_cols = ["encounter_id",'meas_date', 'meas_hour']
cohort_pressors_grouped = cohort_pressors_grouped.groupBy(group_cols) \
                                     .pivot("med_name") \
                                     .agg(f.count('count').alias("count")).orderBy(group_cols)

cohort_pressors_grouped = cohort_pressors_grouped.withColumn("dobutamine_alone", f.expr(
        """
        CASE
        WHEN dobutamine IS NOT NULL AND dopamine IS NULL AND epinephrine IS NULL AND isoproterenol IS NULL AND milrinone IS NULL AND norepinephrine IS NULL AND phenylephrine IS NULL AND vasopressin IS NULL THEN 1
        ELSE 0
        END
        """
    ))

col_list = ['dobutamine', 'dopamine', 'epinephrine', 'isoproterenol', 'milrinone', 'norepinephrine', 'phenylephrine', 'vasopressin']
cohort_pressors_grouped = cohort_pressors_grouped.na.fill(0, subset=col_list)
cohort_pressors_grouped = cohort_pressors_grouped.withColumn('num_pressors', sum([f.col(c) for c in col_list]))

cohort_pressors_grouped = cohort_pressors_grouped.withColumn("num_pressors", f.expr(
        """
        CASE
        WHEN num_pressors > 4 THEN 4
        ELSE num_pressors
        END
        """
    ))
cohort_pressors_grouped = cohort_pressors_grouped.select('encounter_id', 'meas_date', 'meas_hour', 'num_pressors', 'dobutamine_alone')

# Merge back to hourly blocked cohort table 
group_cols = ["encounter_id", "meas_date", "meas_hour"]
pressor_hours = har_ids_hours.join(cohort_pressors_grouped, on=group_cols, how='left')\
        .orderBy(group_cols)\
        .select("encounter_id", "meas_date", "meas_hour", "num_pressors", "dobutamine_alone")\
        .distinct()

# Merge back to resp support hourly table
df = df.join(pressor_hours, on=group_cols, how="left")

# Flag if on life support in an hour
df = df.withColumn("on_life_support", f.expr(
        """
        CASE
        WHEN num_pressors >=1 THEN 1
        WHEN device_filled == 'NIPPV' THEN 1
        WHEN device_filled == 'Vent' THEN 1
        WHEN  p_f < 200 THEN 1
        WHEN s_f < 179 THEN 1
        ELSE 0
        END
        """
    ))

# Create leading flags to identify first episode of 6 consectutive hrs of life support
ls_df = df.orderBy(group_cols)
ls_df = ls_df.withColumn("lead_1", f.lead(f.col("on_life_support"),1).over(Window.partitionBy("encounter_id").orderBy(group_cols)))
ls_df = ls_df.withColumn("lead_2", f.lead(f.col("on_life_support"),2).over(Window.partitionBy("encounter_id").orderBy(group_cols)))
ls_df = ls_df.withColumn("lead_3", f.lead(f.col("on_life_support"),3).over(Window.partitionBy("encounter_id").orderBy(group_cols)))
ls_df = ls_df.withColumn("lead_4", f.lead(f.col("on_life_support"),4).over(Window.partitionBy("encounter_id").orderBy(group_cols)))
ls_df = ls_df.withColumn("lead_5", f.lead(f.col("on_life_support"),5).over(Window.partitionBy("encounter_id").orderBy(group_cols)))

col_list = ['on_life_support', 'lead_1', 'lead_2', 'lead_3', 'lead_4', 'lead_5']

ls_df = ls_df.withColumn('life_support_sum', sum([f.col(c) for c in col_list]))

# Get time first episode of 6 consectutive hrs of life support started
ls_encs = ls_df.filter(f.col('life_support_sum')==6).select('encounter_id', 'meas_date', 'meas_hour')
ls_encs = ls_encs.withColumn("life_support_start", f.concat(ls_encs.meas_date, f.lit(" "), ls_encs.meas_hour))
ls_encs = ls_encs.withColumn('life_support_start',f.to_timestamp('life_support_start','yyyy-MM-dd HH'))
ls_encs = ls_encs.groupBy('encounter_id').agg(f.min('life_support_start').alias("life_support_start"))

ls_encs = ls_encs.withColumn('window_start', (f.col('life_support_start')-f.expr("INTERVAL 42 HOURS")))
ls_encs = ls_encs.withColumn('window_end', (f.col('life_support_start')+f.expr("INTERVAL 5 HOURS")))
ls_encs = ls_encs.select('encounter_id', 'life_support_start', 'window_start', 'window_end')

# Explode between window start and window end to get all hourly timestamps
ls_encs_hours = ls_encs.withColumn('txnDt', f.explode(f.expr('sequence(window_start, window_end, interval 1 hour)')))
ls_encs_hours = ls_encs_hours.withColumn('meas_hour', f.hour(f.col('txnDt')))
ls_encs_hours = ls_encs_hours.withColumn('meas_date', f.to_date(f.col('txnDt')))
ls_encs_hours = ls_encs_hours.select('encounter_id', 'life_support_start', 'meas_date', 'meas_hour', 'window_start', 'window_end').orderBy('encounter_id', 'meas_date', 'meas_hour')

# Re-join age at admission and disposition, admission & discharge dttm, and zip

ls_encs_hours = ls_encs_hours.repartition('encounter_id')
limited_ids = limited_ids.repartition('encounter_id')
demo_disp = demo_disp.repartition('encounter_id')
df = df.repartition('encounter_id')

cohort_blocked = ls_encs_hours.join(limited_ids, on='encounter_id', how='left')
cohort_blocked = cohort_blocked.join(demo_disp, on='encounter_id', how='left')

group_cols = ["encounter_id", "meas_date", "meas_hour"]
cohort_blocked = cohort_blocked.join(df, on=group_cols, how='left')

cohort_blocked.repartition(1).write.csv("sipa_life_support_cohort.csv", mode='overwrite', header="true")
