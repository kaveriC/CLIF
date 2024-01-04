from pyspark.sql import SparkSession
import pyspark.sql.functions as f
from pyspark import SparkConf
from pyspark.sql.window import Window



print("loaded libraries")
spark = SparkSession.builder \
        .appName("SOFA") \
        .getOrCreate()

# Bring in worst vitals. Only need 48 hours for p/f or s/f ratio calculation
vitals = spark.read.parquet("/project2/wparker/SIPA_data/cohort_vitals_48.parquet")


## For SOFA we just need spO2 min
vitals_small = vitals.select('C19_HAR_ID', 'life_support_start_time', 'meas_date', 'meas_hour', 'MAP_for_SOFA')

pf = spark.read.parquet("/project2/wparker/SIPA_data/p_f_combined_filled.parquet")
pf = pf.select('C19_HAR_ID', 'meas_date', 'meas_hour', 'device_filled', 'fio2_filled', 
               'pao2_filled','spo2_filled', 'p_f', 's_f')

group_cols = ['C19_HAR_ID', 'meas_date', 'meas_hour']
vitals_small = vitals_small.join(pf, on=group_cols, how="left")

## bring in meds to get flag for life support yes/no
df_meds = spark.read.option("header",True).csv('/project2/wparker/SIPA_data/RCLIF_meds_admin_conti.csv')
df_meds = df_meds.withColumn('admin_time',f.to_timestamp('admin_time','yyyy-MM-dd HH:mm:ss'))

pressors = df_meds.filter(((f.col('med_name')=='phenylephrine') | 
                       (f.col('med_name')=='epinephrine') | 
                       (f.col('med_name')=='vasopressin') | 
                       (f.col('med_name')=='dopamine') |
                       (f.col('med_name')=='dobutamine') |
                       (f.col('med_name')=='norepinephrine') |
                       (f.col('med_name')=='angiotensin') |
                       (f.col('med_name')=='milrinone') |
                       (f.col('med_name')=='isoproterenol')))
pressors = pressors.select("C19_HAR_ID", "admin_time",'med_name').distinct()

## Get cohort index times
cohort = spark.read.parquet("/project2/wparker/SIPA_data/life_support_cohort.parquet")
cohort = cohort.select('C19_HAR_ID', 'C19_PATIENT_ID', 'adm_date', 'life_support_start', 'disposition_name',
                'sex', 'age_at_adm')
cohort = cohort.withColumnRenamed('life_support_start', 'life_support_start_time')
index_times = cohort.select('C19_HAR_ID', 'life_support_start_time')

## Get all pressors in 48 hour windo
cohort_pressors = index_times.join(pressors,'C19_HAR_ID','left')
cohort_pressors = cohort_pressors.withColumn("hour_diff", (f.col("admin_time").cast("long")-f.col("life_support_start_time").cast("long"))/(60*60))
cohort_pressors_48 = cohort_pressors.filter((f.col('hour_diff')>-42)&(f.col('hour_diff')<=5))
cohort_pressors_48 = cohort_pressors_48.withColumn('meas_hour', f.hour(f.col('admin_time')))
cohort_pressors_48 = cohort_pressors_48.withColumn('meas_date', f.to_date(f.col('admin_time')))

cohort_pressors_48 = cohort_pressors_48.select('C19_HAR_ID','meas_hour','meas_date', 'med_name').distinct()

## Get number of pressors per hour
w2 = Window.partitionBy(group_cols).orderBy("med_name")

group_cols = ['C19_HAR_ID','meas_hour','meas_date', 'med_name']

cohort_pressors_grouped = cohort_pressors_48.withColumn("row",f.row_number().over(w2))
cohort_pressors_grouped = cohort_pressors_grouped.withColumn("dobutamine_alone", f.expr(
        """
        CASE
        WHEN med_name == 'dobutamine' THEN 1
        ELSE 0
        END
        """
    ))

group_cols = ['C19_HAR_ID','meas_hour','meas_date', 'dobutamine_alone']

cohort_pressors_grouped = cohort_pressors_grouped.groupBy(group_cols) \
                                            .agg(f.count('med_name').alias('num_pressors'))

##Flag if on life support at each hour
group_cols = ['C19_HAR_ID','meas_date','meas_hour']
cohort_pressors_vitals = vitals_small.join(cohort_pressors_grouped, on=group_cols, how="left")

cohort_pressors_vitals = cohort_pressors_vitals.withColumn("on_life_support", f.expr(
        """
        CASE
        WHEN num_pressors >=1 THEN 1
        WHEN device_filled == 'NIPPV' THEN 1
        WHEN device_filled == 'Vent' THEN 1
        WHEN device_filled == 'High Flow NC' and p_f < 200 THEN 1
        WHEN device_filled == 'High Flow NC' and p_f < 200 THEN 1
        WHEN device_filled == 'Face Mask' and p_f < 200 THEN 1
        WHEN device_filled == 'Trach Collar' and p_f < 200 THEN 1
        WHEN device_filled == 'Nasal Cannula' and p_f < 200 THEN 1
        WHEN device_filled == 'High Flow NC' and s_f < 179 THEN 1
        WHEN device_filled == 'Face Mask' and s_f < 179 THEN 1
        WHEN device_filled == 'Trach Collar' and s_f < 179 THEN 1
        WHEN device_filled == 'Nasal Cannula' and s_f < 179 THEN 1
        ELSE 0
        END
        """
    ))


cohort_life_support_count = cohort_pressors_vitals.groupBy('C19_HAR_ID') \
                                            .agg(f.sum('on_life_support').alias('life_support_count')) \
                                            .filter(f.col('life_support_count')>=6)

df = cohort_life_support_count.join(cohort_pressors_vitals, on='C19_HAR_ID', how="left")

## add glasgow score
scores = spark.read.option("header",True).csv('/project2/wparker/SIPA_data/RCLIF_scores_10192023.csv')
scores = scores.withColumn('score_time',f.to_timestamp('score_time','yyyy-MM-dd HH:mm:ss'))

cohort_scores = index_times.join(scores,'C19_HAR_ID','left')
cohort_scores = cohort_scores.withColumn("hour_diff", (f.col("score_time").cast("long")-f.col("life_support_start_time").cast("long"))/(60*60))
cohort_scores_48 = cohort_scores.filter((f.col('hour_diff')>-42)&(f.col('hour_diff')<=5))
cohort_scores_48 = cohort_scores_48.withColumn('meas_hour', f.hour(f.col('score_time')))
cohort_scores_48 = cohort_scores_48.withColumn('meas_date', f.to_date(f.col('score_time')))

group_cols = ['C19_HAR_ID','meas_date','meas_hour']
cohort_scores_48 = cohort_scores_48.select('C19_HAR_ID', 'meas_date', 'meas_hour', 'score_name', 'score_value') \
                .distinct() \
                .filter(f.col('score_name')=='NUR RA GLASGOW ADULT SCORING') \
                .groupBy(group_cols) \
                .agg(f.min('score_value').alias('min_gcs_score'))

cohort_scores_48 = cohort_scores_48.select('C19_HAR_ID', 'meas_date', 'meas_hour', 'min_gcs_score')
df = df.join(cohort_scores_48, on=group_cols, how="left")
df = df.withColumn('min_gcs_score', 
                   f.coalesce(f.col('min_gcs_score'),
                              f.last('min_gcs_score', True) \
                              .over(Window.partitionBy('C19_HAR_ID') \
                                    .orderBy(group_cols)), f.lit('NULL')))

## Add labs
labs = spark.read.parquet("/project2/wparker/SIPA_data/cohort_labs_48.parquet")
labs = labs.select('C19_HAR_ID', 'meas_date', 'meas_hour', 'billirubin_max_filled',
                         'platelet_count_min_filled', 'creatinine_max_filled')
df = df.join(labs, on=group_cols, how="left")

## add dialysis flag
dialysis = spark.read.option("header",True).csv('/project2/wparker/SIPA_data/RCLIF_dialysis_flag_only.csv')
dialysis = dialysis.withColumn('recorded_time',f.to_timestamp('recorded_time','yyyy-MM-dd HH:mm:ss'))
dialysis = dialysis.withColumn('meas_hour', f.hour(f.col('recorded_time')))
dialysis = dialysis.withColumn('meas_date', f.to_date(f.col('recorded_time')))

group_cols = ['C19_HAR_ID','meas_date','meas_hour']
cohort_dialysis = index_times.join(dialysis,'C19_HAR_ID','left')
cohort_dialysis = cohort_dialysis.withColumn("hour_diff", (f.col("recorded_time").cast("long")-f.col("life_support_start_time").cast("long"))/(60*60))
cohort_dialysis_48 = cohort_dialysis.filter((f.col('hour_diff')>-42)&(f.col('hour_diff')<=5))

cohort_dialysis_48 = cohort_dialysis_48.select('C19_HAR_ID', 'meas_date', 'meas_hour', 'on_dialysis') \
                .distinct() \
                .groupBy(group_cols) \
                .agg(f.max('on_dialysis').alias('on_dialysis'))
df = df.join(cohort_dialysis_48, on=group_cols, how="left")

df = df.toPandas()
df.to_csv("/project2/wparker/SIPA_data/sipa_df_48.csv")

# Now summarize to 1 obs per person Calculate p/f or s/f ratios
group_cols = ["C19_HAR_ID", 'life_support_start_time']

vitals_small = vitals_small.withColumn("device_rank", f.expr(
        """
        CASE
        WHEN device_filled == 'Vent' THEN 1
        WHEN device_filled == 'NIPPV' THEN 2
        WHEN device_filled == 'High Flow NC' THEN 3
        WHEN device_filled == 'Face Mask' THEN 4 
        WHEN device_filled == 'Trach Collar' THEN 5
        WHEN device_filled == 'Nasal Cannula' THEN 6 
        ELSE NULL
        END
        """
    ))


df_group = vitals_small.groupBy(group_cols) \
        .agg(f.min('spO2_filled').alias('min_spO2'),
             f.min('pao2_filled').alias('min_pao2'),
             f.max('fio2_filled').alias('max_fio2'),
             f.min('p_f').alias('min_PF'),
             f.min('s_f').alias('min_SF'),
             f.max('device_rank').alias('device_rank'))

df_group = df_group.withColumn("max_device", f.expr(
        """
        CASE
        WHEN device_rank == '1' THEN 'Vent'
        WHEN device_rank == '2' THEN 'NIPPV'
        WHEN device_rank == '3' THEN 'High Flow NC'
        WHEN device_rank == '4' THEN 'Face Mask'
        WHEN device_rank == '5' THEN 'Trach Collar'
        WHEN device_rank == '6' THEN 'Nasal Cannula' 
        ELSE NULL
        END
        """
    ))

labs_sum = spark.read.parquet("/project2/wparker/SIPA_data/cohort_labs_48_summary.parquet")
labs_sum = labs_sum.select('C19_HAR_ID', 'life_support_start_time', 'observed_labs_start', 
                            'observed_labs_end', 'billirubin_max_filled',
                         'platelet_count_min_filled', 'creatinine_max_filled')

vitals_sum = spark.read.parquet("/project2/wparker/SIPA_data/cohort_vitals_48_summary.parquet")
vitals_sum = vitals_sum.select('C19_HAR_ID', 'life_support_start_time', 'window_start', 'window_end',
              'observed_vitals_start', 'observed_vitals_end', 'MAP_for_sofa','weight_filled', 'height_filled')

df_group = df_group.repartition('C19_HAR_ID')
labs_sum = labs_sum.repartition('C19_HAR_ID')
vitals_sum = vitals_sum.repartition('C19_HAR_ID')


df_group = df_group.join(labs_sum, on=group_cols, how="full")
df_group_2 = df_group.join(vitals_sum, on=group_cols, how="full")
df_group_3 = cohort.join(df_group_2, on=group_cols, how="left")


cohort_pressors_grouped = cohort_pressors_grouped.groupBy('C19_HAR_ID') \
                                                 .agg((f.min('dobutamine_alone').alias('min_dobutamine_alone')),
                                                     (f.max('num_pressors').alias('max_num_pressors')))
                                                      
df_group_4 = df_group_3.join(cohort_pressors_grouped, on='C19_HAR_ID', how="left")

scores_grouped = cohort_scores_48.groupBy('C19_HAR_ID') \
                                 .agg(f.min('min_gcs_score').alias('min_gcs_score'))
df_group_5 = df_group_4.join(scores_grouped, on='C19_HAR_ID', how="left")

dialysis_grouped = cohort_dialysis_48.groupBy('C19_HAR_ID') \
                        .agg(f.max('on_dialysis').alias('on_dialysis'))
df_group_6 = df_group_5.join(dialysis_grouped, on='C19_HAR_ID', how="left")

df_group_7 = df_group_6.withColumn("sofa_cv", f.expr(
        """
        CASE 
        WHEN max_num_pressors >= 2 THEN 4
        WHEN max_num_pressors == 1 AND min_dobutamine_alone IS NULL THEN 3
        WHEN max_num_pressors == 1 AND min_dobutamine_alone IS NOT NULL THEN 2
        WHEN MAP_for_sofa < 70.0 AND ( max_num_pressors == 0 OR max_num_pressors IS NULL ) THEN 1
        WHEN MAP_for_sofa >= 70.0 AND ( max_num_pressors == 0 OR max_num_pressors IS NULL ) THEN 0
        ELSE NULL
        END
        """
    ))

df_group_7 = df_group_7.withColumn("sofa_coag", f.expr(        
        """
        CASE 
        WHEN platelet_count_min_filled >= 150 OR platelet_count_min_filled IS NULL THEN 0
        WHEN platelet_count_min_filled >= 100 AND platelet_count_min_filled < 150 THEN 1
        WHEN platelet_count_min_filled >= 50 AND platelet_count_min_filled < 100 THEN 2
        WHEN platelet_count_min_filled >= 20 AND platelet_count_min_filled < 50  THEN 3
        WHEN platelet_count_min_filled < 20 THEN 4
        ELSE NULL
        END
        """
    ))

df_group_7 = df_group_7.withColumn("sofa_liver", f.expr(        
        """
        CASE 
        WHEN billirubin_max_filled < 1.2 OR billirubin_max_filled IS NULL THEN 0
        WHEN billirubin_max_filled >=1.2 AND billirubin_max_filled <= 1.9 THEN 1
        WHEN billirubin_max_filled >1.9 AND billirubin_max_filled <= 5.9 THEN 2
        WHEN billirubin_max_filled >=5.9 AND billirubin_max_filled <= 11.9 THEN 3
        WHEN billirubin_max_filled > 12 THEN 4
        ELSE NULL
        END
        """
    ))

df_group_7 = df_group_7.withColumn("sofa_renal", f.expr(        
        """
        CASE 
        WHEN creatinine_max_filled < 1.2 OR creatinine_max_filled IS NULL THEN 0
        WHEN creatinine_max_filled >=1.2 AND creatinine_max_filled < 2 THEN 1
        WHEN creatinine_max_filled >=2 AND creatinine_max_filled < 3.5 THEN 2
        WHEN creatinine_max_filled >=3.5 AND creatinine_max_filled < 5 THEN 3
        WHEN creatinine_max_filled > 5 OR on_dialysis ==1 THEN 4
        ELSE NULL
        END
        """
    ))

df_group_7 = df_group_7.withColumn("sofa_resp_pf", f.expr(        
        """
        CASE 
        WHEN min_PF <= 100 AND max_device IS NOT NULL THEN 4
        WHEN min_PF > 100 AND min_PF <= 200 AND max_device IS NOT NULL THEN 3
        WHEN min_PF > 200 AND min_PF <= 300 THEN 2
        WHEN min_PF > 300 AND min_PF <= 400 THEN 1
        WHEN min_PF > 400 AND max_device != "NIPPV" AND max_device != "Vent" THEN 0
        ELSE NULL
        END
        """
    ))

df_group_7 = df_group_7.withColumn("sofa_resp_sf", f.expr(        
        """
        CASE 
        WHEN min_SF <= 150 AND max_device IS NOT NULL THEN 4
        WHEN min_SF > 150 AND min_SF <= 235 AND max_device IS NOT NULL THEN 3
        WHEN min_SF > 235 AND min_SF <= 315 THEN 2
        WHEN min_SF > 315 AND min_SF <= 400 THEN 1
        WHEN min_SF > 400 AND max_device != "NIPPV" AND max_device != "Vent" THEN 0
        ELSE NULL
        END
        """
    ))

df_group_7 = df_group_7.withColumn("sofa_resp", f.expr(        
        """
        CASE 
        WHEN sofa_resp_pf IS NOT NULL THEN sofa_resp_pf
        WHEN sofa_resp_pf IS NULL AND sofa_resp_sf IS NOT NULL THEN sofa_resp_sf
        ELSE NULL
        END
        """
    ))

df_group_7 = df_group_7.withColumn("sofa_cns", f.expr(        
        """
        CASE 
        WHEN min_gcs_score == 15 THEN 0
        WHEN min_gcs_score >=13 AND min_gcs_score < 15 THEN 1
        WHEN min_gcs_score >=10 AND min_gcs_score < 13 THEN 2
        WHEN min_gcs_score >=6 AND min_gcs_score < 10 THEN 3
        WHEN min_gcs_score <= 5 THEN 4
        ELSE NULL
        END
        """
    ))

df_group_7 = df_group_7.toPandas()
df_group_7.to_csv("/project2/wparker/SIPA_data/sipa_df.csv")
