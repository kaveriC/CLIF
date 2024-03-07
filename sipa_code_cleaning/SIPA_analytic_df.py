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
        .appName("SIPA") \
        .config("spark.driver.memory", "15g") \
        .getOrCreate()

## Read in cohort hourly blocked base dataset
group_cols = ["encounter_id",'meas_date', 'meas_hour']
cohort_hours = spark.read.parquet("/project2/wparker/SIPA_data/life_support_cohort_48.parquet").orderBy(group_cols)


# Bring in worst vitals. Only need MAP for SIPA
vitals = spark.read.parquet("/project2/wparker/SIPA_data/cohort_vitals_48.parquet")
vitals = vitals.select('encounter_id', 'meas_date', 'meas_hour', 'MAP_combined_min')

cohort_resp_vitals = cohort_hours.join(vitals, on=group_cols, how="left").orderBy(group_cols)

## add glasgow score
scores = spark.read.parquet('/project2/wparker/SIPA_data/RCLIF_scores.parquet')
scores = scores.withColumn('score_time',f.to_timestamp('score_time','yyyy-MM-dd HH:mm:ss'))
scores = scores.withColumn('meas_hour', f.hour(f.col('score_time')))
scores = scores.withColumn('meas_date', f.to_date(f.col('score_time')))
scores = scores.withColumn("score_value",scores.score_value.cast('double'))
scores = scores.filter(f.col('score_name')=='NUR RA GLASGOW ADULT SCORING')

## Get min GCS
group_cols = ["encounter_id",'meas_date', 'meas_hour']
scores_wide = scores.groupBy(group_cols) \
                                     .pivot("score_name") \
                                     .agg(f.min('score_value').alias("min")).orderBy(group_cols)
scores_wide = scores_wide.withColumnRenamed('NUR RA GLASGOW ADULT SCORING', 'min_gcs_score')

cohort_scores_48 = cohort_hours.join(scores_wide, on=group_cols, how="left")
cohort_scores_48 = cohort_scores_48.select('encounter_id', 'meas_date', 'meas_hour', 'min_gcs_score')

df = cohort_resp_vitals.join(cohort_scores_48, on=group_cols, how="left")

## Exclude GCS for patients on a ventilator
df = df.withColumn("min_gcs_score_no_vent", f.expr(
        """
        CASE
        WHEN device_filled == "Vent" THEN NULL
        ELSE min_gcs_score
        END
        """
    ))

## Add labs
labs = spark.read.parquet("/project2/wparker/SIPA_data/cohort_labs_48.parquet")
labs = labs.select('encounter_id', 'meas_date', 'meas_hour', 'bilirubin_total_max', 'potassium_max', 'bun_max', 'ph_venous_min',
                         'platelet_count_min', 'creatinine_max', 'carbon_dioxide_min')
df = df.join(labs, on=group_cols, how="left")

## add dialysis flag
dialysis = spark.read.option("header",True).csv('/project2/wparker/SIPA_data/RCLIF_dialysis_flag_only.csv')
dialysis = dialysis.withColumn('recorded_time',f.to_timestamp('recorded_time','yyyy-MM-dd HH:mm:ss'))
dialysis = dialysis.withColumn('meas_hour', f.hour(f.col('recorded_time')))
dialysis = dialysis.withColumn('meas_date', f.to_date(f.col('recorded_time')))
dialysis = dialysis.withColumnRenamed('C19_HAR_ID', 'encounter_id')


dialysis = dialysis.select('encounter_id', 'meas_date', 'meas_hour', 'on_dialysis') \
                .distinct() \
                .groupBy(group_cols) \
                .agg(f.max('on_dialysis').alias('on_dialysis'))
df = df.join(dialysis, on=group_cols, how="left")

df = df.withColumn("bilirubin_total_max",df.bilirubin_total_max.cast('double'))
df = df.withColumn("platelet_count_min",df.platelet_count_min.cast('double'))
df = df.withColumn("creatinine_max",df.creatinine_max.cast('double'))
df = df.withColumn("bun_max",df.bun_max.cast('double'))
df = df.withColumn("ph_venous_min",df.ph_venous_min.cast('double'))
df = df.withColumn("potassium_max",df.potassium_max.cast('double'))
df = df.withColumn("carbon_dioxide_min",df.carbon_dioxide_min.cast('double'))
df = df.withColumn("on_dialysis",df.on_dialysis.cast('double'))

df = df.withColumn("sofa_cv", f.expr(
        """
        CASE 
        WHEN num_pressors >= 2 THEN 4
        WHEN num_pressors == 1 AND dobutamine_alone IS NULL THEN 3
        WHEN num_pressors == 1 AND dobutamine_alone == 0 THEN 3
        WHEN num_pressors == 1 AND dobutamine_alone == 1 THEN 2
        WHEN MAP_combined_min < 70.0 AND ( num_pressors == 0 OR num_pressors IS NULL ) THEN 1
        WHEN MAP_combined_min >= 70.0 AND ( num_pressors == 0 OR num_pressors IS NULL ) THEN 0
        ELSE NULL
        END
        """
    ))

df = df.withColumn("sofa_coag", f.expr(        
        """
        CASE 
        WHEN platelet_count_min >= 150 THEN 0
        WHEN platelet_count_min >= 100 AND platelet_count_min < 150 THEN 1
        WHEN platelet_count_min >= 50 AND platelet_count_min < 100 THEN 2
        WHEN platelet_count_min >= 20 AND platelet_count_min < 50  THEN 3
        WHEN platelet_count_min < 20 THEN 4
        ELSE NULL
        END
        """
    ))

df = df.withColumn("sofa_liver", f.expr(        
        """
        CASE 
        WHEN bilirubin_total_max < 1.2 THEN 0
        WHEN bilirubin_total_max >=1.2 AND bilirubin_total_max <= 1.9 THEN 1
        WHEN bilirubin_total_max >1.9 AND bilirubin_total_max <= 5.9 THEN 2
        WHEN bilirubin_total_max >=5.9 AND bilirubin_total_max <= 11.9 THEN 3
        WHEN bilirubin_total_max > 12 THEN 4
        ELSE NULL
        END
        """
    ))

df = df.withColumn("sofa_renal", f.expr(        
        """
        CASE 
        WHEN creatinine_max < 1.2 THEN 0
        WHEN creatinine_max >=1.2 AND creatinine_max < 2 THEN 1
        WHEN creatinine_max >=2 AND creatinine_max < 3.5 THEN 2
        WHEN creatinine_max >=3.5 AND creatinine_max < 5 THEN 3
        WHEN creatinine_max > 5 OR on_dialysis ==1 THEN 4
        ELSE NULL
        END
        """
    ))

df = df.withColumn("sofa_resp_pf", f.expr(        
        """
        CASE 
        WHEN p_f <= 100 AND device_filled IS NOT NULL THEN 4
        WHEN p_f > 100 AND p_f <= 200 AND device_filled IS NOT NULL THEN 3
        WHEN p_f > 200 AND p_f <= 300 THEN 2
        WHEN p_f > 300 AND p_f <= 400 THEN 1
        WHEN p_f > 400 AND device_filled != "NIPPV" AND device_filled != "Vent" THEN 0
        ELSE NULL
        END
        """
    ))

df = df.withColumn("sofa_resp_sf", f.expr(        
        """
        CASE 
        WHEN s_f <= 150 AND device_filled IS NOT NULL THEN 4
        WHEN s_f > 150 AND s_f <= 235 AND device_filled IS NOT NULL THEN 3
        WHEN s_f > 235 AND s_f <= 315 THEN 2
        WHEN s_f > 315 AND s_f <= 400 THEN 1
        WHEN s_f > 400 AND device_filled != "NIPPV" AND device_filled != "Vent" THEN 0
        ELSE NULL
        END
        """
    ))

df = df.withColumn("sofa_resp", f.expr(        
        """
        CASE 
        WHEN sofa_resp_pf IS NOT NULL THEN sofa_resp_pf
        WHEN sofa_resp_pf IS NULL AND sofa_resp_sf IS NOT NULL THEN sofa_resp_sf
        ELSE NULL
        END
        """
    ))

df = df.withColumn("sofa_cns", f.expr(        
        """
        CASE 
        WHEN min_gcs_score_no_vent == 15 THEN 0
        WHEN min_gcs_score_no_vent >=13 AND min_gcs_score_no_vent < 15 THEN 1
        WHEN min_gcs_score_no_vent >=10 AND min_gcs_score_no_vent < 13 THEN 2
        WHEN min_gcs_score_no_vent >=6 AND min_gcs_score_no_vent < 10 THEN 3
        WHEN min_gcs_score_no_vent <= 5 THEN 4
        ELSE NULL
        END
        """
    ))

df = df.distinct()
df.repartition(1).write.csv("/project2/wparker/SIPA_data/sipa_df_48.csv", mode='overwrite', header="true")

# Now summarize to 1 obs per person 
df = df.withColumn("device_rank", f.expr(
        """
        CASE
        WHEN device_filled == 'Vent' THEN 1
        WHEN device_filled == 'NIPPV' THEN 2
        WHEN device_filled == 'CPAP' THEN 3
        WHEN device_filled == 'High Flow NC' THEN 4
        WHEN device_filled == 'Face Mask' THEN 5
        WHEN device_filled == 'Trach Collar' THEN 6
        WHEN device_filled == 'Nasal Cannula' THEN 7
        WHEN device_filled == 'Other' THEN 8
        WHEN device_filled == 'Room Air' THEN 9
        WHEN device_filled IS NULL THEN NULL
        ELSE NULL
        END
        """
    ))

group_cols = ["encounter_id", "patient_id", "zip_code", "admission_dttm", "discharge_dttm", "life_support_start", "window_start", "window_end", "age_at_admission", "disposition"]

df_sum= df.groupBy(group_cols) \
        .agg(f.min('spO2').alias('min_spO2'),
             f.min('pao2_filled').alias('min_pao2'),
             f.max('fio2_filled').alias('max_fio2'),
             f.min('p_f').alias('min_PF'),
             f.min('s_f').alias('min_SF'),
             f.min('device_rank').alias('device_rank'),
             f.max('num_pressors').alias('max_num_pressors'),
             f.min('dobutamine_alone').alias('min_dobutamine_alone'),
             f.max('dobutamine_alone').alias('max_dobutamine_alone'),
             f.min('MAP_combined_min').alias('min_MAP'),
             f.min('min_gcs_score_no_vent').alias('min_gcs_score'),
             f.max('bilirubin_total_max').alias('max_bilirubin'),
             f.max('creatinine_max').alias('max_creatinine'),
             f.min('platelet_count_min').alias('min_platelet'),
             f.max('potassium_max').alias('max_potassium'),
             f.max('bun_max').alias('max_bun'),
             f.min('carbon_dioxide_min').alias('min_carbon_dioxide'),
             f.min('ph_venous_min').alias('min_ph_venous'),
             f.max('on_dialysis').alias('on_dialysis'),
             f.max('sofa_cv').alias('max_sofa_cv'),
             f.max('sofa_coag').alias('max_sofa_coag'),
             f.max('sofa_liver').alias('max_sofa_liver'),
             f.max('sofa_renal').alias('max_sofa_renal'),
             f.max('sofa_resp_pf').alias('max_sofa_resp_pf'),
             f.max('sofa_resp_sf').alias('max_sofa_resp_sf'),
             f.max('sofa_resp').alias('max_sofa_resp'),
             f.max('sofa_cns').alias('max_sofa_cns')).distinct()

df_sum = df_sum.withColumn("max_device_name", f.expr(
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

df_sum.repartition(1).write.csv("/project2/wparker/SIPA_data/sipa_df_summary.csv", mode='overwrite', header="true")

