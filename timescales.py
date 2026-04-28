"""Author: Pratitee & Aditya
Last Update: 22/09/2025"""

# !pip install openpyxl -q
 
import warnings 
import os, sys
import json 
import re
import numpy as np
from datetime import datetime, timedelta
import pandas as pd
from pyspark.errors import PySparkException
from pyspark.sql.types import StructType, StructField, BooleanType, StringType, IntegerType , DateType
from pyspark.sql.functions import from_json, col, lit, udf, when, expr
from pyspark.sql.functions import lower
from pyspark.sql.functions import date_format
from pyspark.sql import functions as F
from pyspark.sql.types import (
    StringType, TimestampType, DateType, DoubleType,
    IntegerType, BooleanType
)
from prompts.timescales_prompt import system_prompt_refund_timescales , system_prompt_delivery_prompt , system_prompt_payment_prompt, system_prompt_cancellation_refund_timeline, refund_promise_prompt
import yaml
from pyspark.sql import functions as F
sys.path.append('../')
from utils.iaudit_logger import get_logger
from databricks.sdk.runtime import *
warnings.filterwarnings("ignore")

main_config = yaml.safe_load(open('../main_config.yaml', 'r'))
llama_8b = main_config['LLM']['llama']

response_format = json.dumps({
    "type": "json_schema",
    "json_schema": {
        "name": "refund_or_adjustment_detection",
        "schema": {
            "type": "object",
            "properties": {
                "refund_or_adjustment_made": {"type": "string", "enum": ["Yes", "No"]},
                "type_of_adjustment": {"type": ["string", "null"], "enum": ["Refund", "Adjustment", "Both", None]},
                "refund_timeline_provided": {"type": "string", "enum": ["Yes", "No"]},
                "refund_timeline_text": {"type": ["string", "null"]},
                "score": {"type": "integer", "enum": [0, 1, 10]}
            },
            "required": ["refund_or_adjustment_made", "type_of_adjustment", "refund_timeline_provided", "refund_timeline_text", "score"]
           
        },
         "strict": True
    }
})


response_format_delivery = json.dumps({
    "type": "json_schema",
    "json_schema": {
        "name": "delivery_promise_detection",
        "schema": {
            "type": "object",
            "properties": {
                "delivery_promised": {"type": "boolean"},
                "delivery_date_provided": {"type": "boolean"},
                "delivery_date_text": {"type": ["string", "null"]}
            },
            "required": ["delivery_promised", "delivery_date_provided", "delivery_date_text"],
            "strict": True
        }
    }
})

response_format_payment = json.dumps({
    "type": "json_schema",
    "json_schema": {
        "name": "payment_detection",
        "schema": {
            "type": "object",
            "properties": {
                "payment_related": {"type": "boolean"},
                "payment_type": {
                    "type": ["string", "null"],
                    "enum": ["missing_payment", "make_payment", None]
                },
                "payment_method": {"type": ["string", "null"]},
                "payment_timeline_provided": {"type": "boolean"},
                "payment_timeline_text": {"type": ["string", "null"]}
            },
            "required": [
                "payment_related",
                "payment_type",
                "payment_method",
                "payment_timeline_provided",
                "payment_timeline_text"
            ],
            "strict": True
        }
    }
})

response_format_cancellation = json.dumps({
    "type": "json_schema",
    "json_schema": {
        "name": "cancellation_refund_timeline_detection",
        "schema": {
            "type": "object",
            "properties": {
                "cancellation_requested": {"type": "boolean"},
                "refund_timeline_provided": {"type": ["boolean", "null"]},
                "refund_timeline_text": {"type": ["string", "null"]}
            },
            "required": ["cancellation_requested", "refund_timeline_provided", "refund_timeline_text"],
            "strict": True
        }
    }
})


def group(df,logger):
    try:
        df = df[['callid', 'transcript', 'channel', 'AgentId', 'createdate']]
        df['pk'] = df['callid'] + "|" + df['AgentId']
        df['transcriptn'] = df['channel'] + ':' + df['transcript']
        convo_df = df.groupby('pk').agg({
            'transcriptn': lambda x: '\n'.join(x),
            'createdate': 'first'
        }).reset_index()
        convo_df.columns = ['pk', 'transcript', 'createdate']
        convo_df['callid'] = convo_df['pk'].apply(lambda x: x.split('|')[0])
        return convo_df
    except Exception as e:
        logger.error("'Timescales-Module': Error in 'group' function: %s", e)
        raise



def comments_results(df, logger):
    """Adding Comments and Results to the Dataframe based on multiple score columns, including cancellation"""
    try:
        # List of score columns to consider
        score_cols = ["score_refund", "score_delivery", "score_payment", "score_cancellation"]

        # Compute score_final_timescales
        df = df.withColumn(
            "score_final_timescales",
            when(
                (col("score_refund") == 0) | (col("score_delivery") == 0) | (col("score_payment") == 0) | (col("score_cancellation") == 0), lit(0)
            ).when(
                (col("score_refund") == 1) | (col("score_delivery") == 1) | (col("score_payment") == 1) | (col("score_cancellation") == 1), lit(1)
            ).otherwise(lit(10))
        )

        # Comments logic
        df = df.withColumn(
            "comments_timescales",
            when(
                (col("score_final_timescales") == 0) & (col("score_refund") == 0),
                lit("Incorrect refund timescale - less than 1 working day")
            ).when(
                (col("score_final_timescales") == 0) & (col("score_delivery") == 0),
                lit("Incorrect delivery timescale - less than expected date")
            ).when(
                (col("score_final_timescales") == 0) & (col("score_payment") == 1) & (col("payment_timeline_provided") == False),
                lit("Payment timeline not provided")
            ).when(
                (col("score_final_timescales") == 0) & (col("score_payment") == 0) & (col("score_cancellation") == 0),
                lit("Incorrect payment timescale - less than allowed days")
            ).when(
                (col("score_final_timescales") == 0) & (col("score_cancellation") == 0),
                lit("Incorrect cancellation refund timescale - less than allowed days")
            ).when(
                (col("score_final_timescales") == 1) & (col("score_refund") == 1),
                lit("Incorrect refund timescale - more than 1 working day")
            ).when(
                (col("score_final_timescales") == 1) & (col("score_delivery") == 1),
                lit("Incorrect delivery timescale - more than expected date")
            ).when(
                (col("score_final_timescales") == 1) & (col("score_payment") == 1),
                lit("Incorrect payment timescale - more than allowed days")
            ).when(
                (col("score_final_timescales") == 1) & (col("score_cancellation") == 1),
                lit("Incorrect cancellation refund timescale - more than allowed days")
            ).otherwise(lit(None))
        )
        

        # Results logic
        df = df.withColumn(
            "results_timescales",
            when(col("score_final_timescales") == 0, lit("Poor outcome"))
            .when(col("score_final_timescales") == 1, lit("No Poor outcome"))
            .otherwise(lit("Compliant - Good Outcome"))
        )
        return df
    except Exception as e:
        logger.error(f"'Timescales-Module':Error in comments_results function: %s", e)
        raise



def timescales_process(df_process, df1_process, logger):
    """Timescales Processing via Logic and LLM"""
    try:
        system_prompt_timescales_json = json.dumps(system_prompt_refund_timescales)

        df_process = df_process.withColumn("transcript", col("transcript").cast("string"))
        display(spark.createDataFrame([{"df_process_count": df_process.count()}]))

        df1_sel = df1_process.select("callid", "Level_2", "CallType", "call_date", "account_type","account_no")
        df_merged = df_process.join(df1_sel, on="callid", how="left")
        display(spark.createDataFrame([{"df_merged_count": df_merged.count()}]))
        display(spark.createDataFrame([{
            "num_callids_with_na": df1_process.filter(
                (col("Level_2").isNull()) | (col("CallType").isNull())
            ).select("callid").distinct().count()
        }]))

        level2_filter_values = [
            "Delivery Charge", "Collection Charge","INR - Courier Not Billed",
            "INR - Courier Billed",
            "INR - Store Billed",
            "INR - Store Not Billed",
            "INR - Upholstery - Billed",
            "INR - Upholstery Not Billed",
            "INR - Wrong Address/Store Billed",
            "INR - Wrong Address/Store Not Billed",
            "Delivery Charge",
            "Collection Charge",
            "Returns Not Credited - Store ",
            "Returns Not Credited - Courier",
            "Refund - Not Received",
            "Not In Parcel Billed",
            "Not in Parcel Not Billed",
            "Returns not credited - Supplier ",
            "INR - Parcel shop Billed",
            "INR - Parcel Shop Not Billed",
            "Returns Not Credited - Carrier",
            "Returns Not Credited - Evri Locker",
            "Returns Not Credited - Parcelshop",
            "Returns Not Credited - Post",
            "Returns Not Credited - Store ",

            "Refund - Not Received",
            "Returns Not Credited - Carrier",
            "Returns Not Credited - Courier",
            "Returns Not Credited - Evri Locker",
            "Returns Not Credited - Parcelshop",
            "Returns Not Credited - Post",
            "Returns Not Credited - Store",
            "Returns Not Credited - Supplier",
            "GNR - 3rd Party Supplier - Billed",
            "GNR - 3rd Party Supplier Not Billed",
            "GNR - Courier Billed",
            "GNR - Courier Not Billed",
            "GNR - Failed Delivery Billed",
            "GNR - Failed Delivery Not Billed",
            "GNR - International Carrier",
            "GNR - Parcel shop Billed",
            "GNR - Parcel Shop Not Billed",
            "GNR - Store Billed",
            "GNR - Store Not Billed",
            "GNR - Upholstery - Billed",
            "GNR - Upholstery Not Billed",
            "GNR - Wrong Address/Store Billed",
            "GNR - Wrong Address/Store Not Billed",
            "Not In Parcel Billed",
            "Not in Parcel Not Billed",
            "Interest charge"
        ]

        df_filtered = df_merged.filter(col("Level_2").isin(level2_filter_values))
        df_rest = df_merged.subtract(df_filtered)


        display(spark.createDataFrame([{"df_filtered": df_filtered.count()}]))
        display(spark.createDataFrame([{"df_rest": df_rest.count()}]))

        result_df = df_filtered.withColumn("result", expr(f"ai_query('{llama_8b}', request => concat({system_prompt_timescales_json}, transcript, '\nCall date:\n', call_date), responseFormat => '{response_format}')"))

        response_schema = StructType([
            StructField("refund_or_adjustment_made", StringType(), True),
            StructField("type_of_adjustment", StringType(), True),
            StructField("refund_timeline_provided", StringType(), True),
            StructField("refund_timeline_text", StringType(), True),
            StructField("score", IntegerType(), True)
        ])

        df_parsed = result_df.withColumn("response_parsed", from_json(col("result"), response_schema))

        df_final = df_parsed.select(
            "*",
            col("response_parsed.refund_or_adjustment_made").alias("refund_or_adjustment_made"),
            col("response_parsed.type_of_adjustment").alias("type_of_adjustment"),
            col("response_parsed.refund_timeline_provided").alias("refund_timeline_provided"),
            col("response_parsed.refund_timeline_text").alias("refund_timeline_text"),
            col("response_parsed.score").alias("score")
        ).drop("response_parsed", "result")

        # Set score=10 and other columns blank if Level_2 is null in df_final
        df_final = df_final.withColumn(
            "score",
            when(col("Level_2").isNull(), lit(10)).otherwise(col("score"))
        ).withColumn(
            "refund_or_adjustment_made",
            when(col("Level_2").isNull(), lit("")).otherwise(col("refund_or_adjustment_made"))
        ).withColumn(
            "type_of_adjustment",
            when(col("Level_2").isNull(), lit("")).otherwise(col("type_of_adjustment"))
        ).withColumn(
            "refund_timeline_provided",
            when(col("Level_2").isNull(), lit("")).otherwise(col("refund_timeline_provided"))
        ).withColumn(
            "refund_timeline_text",
            when(col("Level_2").isNull(), lit("")).otherwise(col("refund_timeline_text"))
        )



        # Add required columns to df_rest with NA (None) and score=10
        df_rest_aug = df_rest.withColumn("refund_or_adjustment_made", lit(None).cast(StringType())) \
            .withColumn("type_of_adjustment", lit(None).cast(StringType())) \
            .withColumn("refund_timeline_provided", lit(None).cast(StringType())) \
            .withColumn("refund_timeline_text", lit(None).cast(StringType())) \
            .withColumn("score", lit(10).cast(IntegerType()))

        # Concatenate
        df_concat = df_final.unionByName(df_rest_aug)
        df_concat = df_concat.withColumnRenamed("score", "score_refund")
        df_concat = df_concat.withColumn(
            "comments_refund",
            when(col("score_refund") == 0, lit("Incorrect refund timescale - less than 1 working day"))
            .when(col("score_refund") == 1, lit("Incorrect refund timescale - more than 1 working day"))
            .otherwise(lit(None))
        )
        print("Refund timeline llm done pandas manipulation")
        pandas_df = df_concat.toPandas()
        pandas_df['comments_refund'] = np.where((~pandas_df['Level_2'].isin(level2_filter_values)), 'N.A. Level2' ,pandas_df['comments_refund'] )

        pandas_df['comments_refund'] = np.where((pandas_df['refund_or_adjustment_made'] != 'Yes')&(pandas_df['Level_2'].isin(level2_filter_values)), 'N.A. No refund discussed',pandas_df['comments_refund'] )

        # initialization & BOD
        pandas_df['results_refund'] = 'Compliant - Good Outcome'
        pandas_df['score_refund'] = pandas_df['score_refund'].fillna(10)

        pandas_df['results_refund'] = np.where(pandas_df['comments_refund'].str.contains("Incorrect refund timescale - more than 1 working day"), 'Compliant with Development',pandas_df['results_refund'] )

        pandas_df['results_refund'] = np.where(pandas_df['comments_refund'].str.contains("Incorrect refund timescale - less than 1 working day"), 'No Poor outcome',pandas_df['results_refund'] )

        pandas_df['score_refund'] = np.where(pandas_df['comments_refund'].str.contains("Incorrect refund timescale - less than 1 working day"), 1 ,pandas_df['score_refund'] )

        one_working_day = ["one working", "24 hours", "tomorrow", "one day", "1 working day", "1 day", "next day", "next working"]
        pattern_one_working_day = "|".join(one_working_day)
        pandas_df['positive_one_day'] =pandas_df['refund_timeline_text'].str.contains(pattern_one_working_day, case= False, na = False )
        pandas_df['results_refund'] = np.where((pandas_df['refund_timeline_text'].str.contains(pattern_one_working_day, case= False, na = False  )), 'Compliant - Good Outcome', pandas_df['results_refund'] )

        pandas_df['score_refund'] = np.where((pandas_df['refund_timeline_text'].str.contains(pattern_one_working_day, case= False, na = False  )), 10 , pandas_df['score_refund'] )

        pandas_df['comments_refund'] = np.where((pandas_df['refund_timeline_text'].str.contains(pattern_one_working_day, case= False, na = False  )), '', pandas_df['comments_refund'] )

        #cash account_type
        print("cash account type")
        list_payment_provider = ["depends on your payment provider", "payment provider", "payments provider", "banking provider", "bank provider", "provider", "banking partner", "payments partner", "payment partner", "depends on your bank","check with your bank", "bank", "dependent on your payment provided", "depends on", "dependent on"]
        pattern_payment_provider = "|".join(list_payment_provider)

        pandas_df['results_refund'] = np.where(((pandas_df['account_type'].str.lower() == 'cash')&(pandas_df['score_refund']==10)&(pandas_df['type_of_adjustment'].str.lower() == 'refund')&(~(pandas_df['transcript'].str.contains(pattern_payment_provider, case= False, na = False )))), 'Compliant with Development' ,pandas_df['results_refund'])

        pandas_df['comments_refund'] = np.where(((pandas_df['account_type'].str.lower() == 'cash')&(pandas_df['score_refund']==10)&(pandas_df['type_of_adjustment'].str.lower() == 'refund')&(~(pandas_df['transcript'].str.contains(pattern_payment_provider, case= False, na = False )))), 'Cash account- timescale depends on payment provider not mentioned' ,pandas_df['comments_refund'])

        pandas_df['score_refund'] = np.where(pandas_df['comments_refund'] == 'Cash account- timescale depends on payment provider not mentioned', 1, pandas_df['score_refund'])

        print("Refund timeline done")

        ###cash account type ends





        pandas_df.to_excel('./intermediate_refund_timescale.xlsx')

        df_concat = spark.createDataFrame(pandas_df)
        # df_comments = comments_results(df_concat, logger)
        
        # df_comments = df_comments.withColumn(
        #     "score",
        #     when(col("results") == "Compliant - Good Outcome", lit(10)).otherwise(col("score"))
        # )

        return df_concat
    except PySparkException as ex:
        if ex.getErrorClass() == "TABLE_OR_VIEW_NOT_FOUND":
            logger.error("'Timescales-Module': 'timescales_process': Table or view not found: " + ex.getMessageParameters()['relationName'])
        else:
            logger.error("'Timescales-Module':'timescales_process': PySparkException in timescales_process: ", ex)
        raise
    except Exception as e:
        logger.error("'Timescales-Module':Error in timescales_process function: ", e)
        raise


def validate_refund_markdowns(dfspark):
    validation_prompt = json.dumps(refund_promise_prompt)
    ai_query_expr =f"ai_query('{llama_8b}',request => concat('{validation_prompt}', '\n', transcript))"
    dfspark = dfspark.withColumn(
        "refund_val_LLM",
        expr(f"CASE WHEN score_refund IN (0, 1) THEN {ai_query_expr} ELSE NULL END")
    )
    
    print("Refund validation llm done post processing is being done")
    # Define schema for the JSON structure in refund_val_LLM
    refund_val_schema = StructType([
        StructField("Refund initiated", StringType(), True),
        StructField("Refund initiated evidence", StringType(), True),
        StructField("Refund reason", StringType(), True),
        StructField("Refund Timeline", StringType(), True)
        
    ])

    # Convert string to JSON struct
    dfspark = dfspark.withColumn("refund_val_LLM_json", from_json(col("refund_val_LLM"), refund_val_schema))

    # Extract fields from JSON
    dfspark = dfspark \
        .withColumn("refund_initiated", col("refund_val_LLM_json.`Refund initiated`")) \
        .withColumn("refund_initiated_evidence", col("refund_val_LLM_json.`Refund initiated evidence`")) \
        .withColumn("refund_reason", col("refund_val_LLM_json.`Refund reason`"))\
        .withColumn("refund_timeline", col("refund_val_LLM_json.`Refund Timeline`"))   


    dfspark = dfspark.withColumn(
        "comments_refund",
        when((col("refund_timeline") == "False"), lit("Advisor failed to inform customer the refund timeline") ).otherwise(col("comments_refund"))
    )

    dfspark = dfspark.withColumn(
        "score_refund",
        when((col("refund_timeline") == "False"), 1 ).otherwise(col("score_refund"))
    )

    dfspark = dfspark.withColumn(
        "results_refund",
        when((col("refund_timeline") == "False"), lit("Compliant with Development") ).otherwise(col("results_refund"))
    )

    dfspark = dfspark.withColumn(
        "score_refund",
        when((col("refund_initiated") == "False") | (lower(col("refund_reason")).contains("cancel")), 10).otherwise(col("score_refund"))
    )

    dfspark = dfspark.withColumn(
        "comments_refund",
        when((col("refund_initiated") == "False") | (lower(col("refund_reason")).contains("cancel")), lit("")).otherwise(col("comments_refund"))
    )

    # cols_to_remove = ["refund_initiated", "refund_val_LLM_json", "refund_initiated_evidence", "refund_reason"]
    # dfspark = dfspark.drop(*cols_to_remove)

    dfspark = dfspark.withColumn(
        "results_refund",
        when(col("score_refund") == 10, lit("Compliant - Good Outcome")).otherwise(col("results_refund"))
    )

    print("refund validation done")

    return dfspark


def _parse_delivery_date(text: str, ref_date_str: str):
    if not text or not ref_date_str:
        return None
    txt = text.lower()
    try:
        ref_dt = datetime.strptime(ref_date_str, "%d/%m/%Y")
    except Exception:
        try:
            ref_dt = datetime.strptime(ref_date_str, "%Y-%m-%d")
        except Exception:
            return None
    if re.search(r"\btoday\b", txt):
        return ref_dt
    if re.search(r"\btomorrow\b", txt):
        return ref_dt + timedelta(days=1)
    m = re.search(r"in\s+(\d+)\s+day", txt)
    if m:
        days = int(m.group(1))
        return ref_dt + timedelta(days=days)
    weekdays = ["monday", "tuesday", "wednesday", "thursday", "friday", "saturday", "sunday"]
    for i, wd in enumerate(weekdays):
        if re.search(rf"\b{wd}\b", txt):
            days_ahead = (i - ref_dt.weekday() + 7) % 7
            days_ahead = days_ahead if days_ahead != 0 else 7
            return ref_dt + timedelta(days=days_ahead)
    m = re.search(r"by\s+(" + "|".join(weekdays) + r")", txt)
    if m:
        wd = m.group(1)
        i = weekdays.index(wd)
        days_ahead = (i - ref_dt.weekday() + 7) % 7
        days_ahead = days_ahead if days_ahead != 0 else 7
        return ref_dt + timedelta(days=days_ahead)
    # Try to parse explicit date in dd/mm/yyyy or dd-mm-yyyy
    m = re.search(r"(\d{1,2})[/-](\d{1,2})[/-](\d{2,4})", txt)
    if m:
        day, month, year = m.groups()
        year = year if len(year) == 4 else "20" + year.zfill(2)
        try:
            return datetime.strptime(f"{day}/{month}/{year}", "%d/%m/%Y")
        except Exception:
            return None
    return None

parsed_payment_datee_udf = udf(_parse_delivery_date, DateType())

def add_payment_expected_date_features(df):
    df = df.withColumn("call_date", F.to_date("createdate"))
    df = df.withColumn("dow", F.date_format("createdate", "E"))
    df = df.withColumn("hour", F.hour("createdate"))
    df = df.withColumn("is_after_10pm", F.col("hour") >= 22)
    df = df.withColumn(
        "timeline_provided",
        parsed_payment_datee_udf(F.col("payment_timeline_text"), F.date_format(F.col("call_date"), "dd/MM/yyyy"))
    )
    df = df.withColumn(
        "calculated_expected_date",
        F.when(
            (F.col("dow") == "Mon") & (~F.col("is_after_10pm")),
            F.date_add(F.to_date("createdate"), 1)
        ).when(
            (F.col("dow") == "Mon") & (F.col("is_after_10pm")),
            F.date_add(F.to_date("createdate"), 2)
        ).when(
            (F.col("dow") == "Tue") & (~F.col("is_after_10pm")),
            F.date_add(F.to_date("createdate"), 1)
        ).when(
            (F.col("dow") == "Tue") & (F.col("is_after_10pm")),
            F.date_add(F.to_date("createdate"), 2)
        ).when(
            (F.col("dow") == "Wed") & (~F.col("is_after_10pm")),
            F.date_add(F.to_date("createdate"), 1)
        ).when(
            (F.col("dow") == "Wed") & (F.col("is_after_10pm")),
            F.date_add(F.to_date("createdate"), 2)
        ).when(
            (F.col("dow") == "Thu") & (~F.col("is_after_10pm")),
            F.date_add(F.to_date("createdate"), 1)
        ).when(
            (F.col("dow") == "Thu") & (F.col("is_after_10pm")),
            F.date_add(F.to_date("createdate"), 2)
        ).when(
            (F.col("dow") == "Fri") & (~F.col("is_after_10pm")),
            F.date_add(F.to_date("createdate"), 1)
        ).when(
            (F.col("dow") == "Fri") & (F.col("is_after_10pm")),
            F.date_add(F.to_date("createdate"), 4)
        ).when(
            F.col("dow") == "Sat",
            F.date_add(F.to_date("createdate"), 3)
        ).when(
            F.col("dow") == "Sun",
            F.date_add(F.to_date("createdate"), 2)
        )
    )
    df = df.withColumn(
        "provided_correct_timeline",
        F.to_date("timeline_provided") == F.col("calculated_expected_date")
    )

    return df

def delivery_process(df_process, df1_process, logger):
    """Delivery Processing via Logic and LLM"""
    try:
        system_prompt_delivery_json = json.dumps(system_prompt_delivery_prompt)
        df_process = df_process.withColumn("transcript", col("transcript").cast("string"))
        df1_sel = df1_process.select("callid", "Level_2", "CallType", "call_date", "exp_delv_date")
        df_merged = df_process.join(df1_sel, on="callid", how="left")
        level2_filter_values = [
            "Place an order",
            "Parcel status check",
            "Checking Delivery date or time",
            "Amend delivery",
            "Book delivery",
            "Delivery timeframe",
            "Delivery method/option",
            "Held order update"
        ]
        df_merged = df_merged.withColumn("Level_2", col("Level_2").cast("string"))
        df_merged = df_merged.withColumn("Level_2", lower(col("Level_2")))
        level2_filter_values_lower = [v.lower() for v in level2_filter_values]
        df_filtered = df_merged.filter(col("Level_2").isin(level2_filter_values_lower))
        df_rest = df_merged.subtract(df_filtered)
        result_df = df_filtered.withColumn(
            "result",
            expr(f"ai_query('{llama_8b}', request => concat({system_prompt_delivery_json}, transcript), responseFormat => '{response_format_delivery}')")
        )
        #
        response_schema = StructType([
            StructField("delivery_promised", BooleanType(), True),
            StructField("delivery_date_provided", BooleanType(), True),
            StructField("delivery_date_text", StringType(), True)
        ])
        df_parsed = result_df.withColumn("response_parsed", from_json(col("result"), response_schema))
        df_final = df_parsed.select(
            "*",
            col("response_parsed.delivery_promised").alias("delivery_promised"),
            col("response_parsed.delivery_date_provided").alias("delivery_date_provided"),
            col("response_parsed.delivery_date_text").alias("delivery_date_text")
        ).drop("response_parsed", "result")
        # -------------------------------------------------
        # Parse relative delivery dates from `delivery_date_text`
        # -------------------------------------------------
        def _to_datetime(date_str):
            try:
                return datetime.strptime(date_str, "%d/%m/%Y")
            except Exception:
                return None
        # Registers the _parse_delivery_date Python function as a Spark UDF (User Defined Function) that returns a DateType, so it can be used to transform columns in a Spark DataFrame.
        parse_delivery_date_udf = udf(_parse_delivery_date, DateType())
        # display(df_final.columns)
        df_final = df_final.withColumn(
            "extracted_date",
            parse_delivery_date_udf(col("delivery_date_text"), date_format(col("call_date"), "dd/MM/yyyy"))
        )
        # display(df_final.columns)
        #
        # df_final = df_final.withColumn(
        #     "extracted_date",
        #     parse_delivery_date_udf(col("delivery_date_text"), col("call_date"))
        # )
        # df_final = df_final.withColumn(
        #     "extracted_date_formatted",
        #     date_format(col("extracted_date"), "dd/MM/yyyy")
        # )
        #
        df_final = df_final.withColumn(
            "score_delivery",
            when(col("extracted_date") > col("exp_delv_date"), lit(1))
            .when(col("extracted_date") < col("exp_delv_date"), lit(0))
            .otherwise(lit(10))
        )
        #
        df_final = df_final.withColumn(
            "delivery_promised",
            when(col("Level_2").isNull(), lit(None)).otherwise(col("delivery_promised"))
        ).withColumn(
            "delivery_date_provided",
            when(col("Level_2").isNull(), lit(None)).otherwise(col("delivery_date_provided"))
        ).withColumn(
            "delivery_date_text",
            when(col("Level_2").isNull(), lit("")).otherwise(col("delivery_date_text"))
        )
        #
        df_rest_aug = df_rest.withColumn("delivery_promised", lit(None).cast(BooleanType())) \
            .withColumn("delivery_date_provided", lit(None).cast(BooleanType())) \
            .withColumn("delivery_date_text", lit(None).cast(StringType())) \
            .withColumn("extracted_date", lit(None).cast(DateType())) \
            .withColumn("score_delivery", lit(10).cast(IntegerType()))
        #
        df_concat = df_final.unionByName(df_rest_aug)
        #
        return df_concat
    except PySparkException as ex:
        if ex.getErrorClass() == "TABLE_OR_VIEW_NOT_FOUND":
            logger.error("'Delivery-Module': 'delivery_process': Table or view not found: " + ex.getMessageParameters()['relationName'])
        else:
            logger.error("'Delivery-Module':'delivery_process': PySparkException in delivery_process: ", ex)
        raise
    except Exception as e:
        logger.error("'Delivery-Module':Error in delivery_process function: ", e)
        raise

def payment_process(df_process, df1_process, logger):
    """
    Payment Processing via Logic and LLM.
    - Divide dataset by Level_2: 'Make payment' or 'Missing payment' vs rest.
    - Scoring logic:
        - If payment_type == 'make_payment' and payment_method == 'card payment': score = 1 if parsed date <= call_date + 1 working day, else 0.
        - If payment_type == 'missing_payment' and payment_method in ['bank transfer', 'internet banking']: score = 1 if parsed date <= call_date + 5 working days, else 0.
        - Else, score = 10.
    """
    try:
        system_prompt_payments_json = json.dumps(system_prompt_payment_prompt)
        df_process = df_process.withColumn("transcript", col("transcript").cast("string"))
        df1_sel = df1_process.select("callid", "Level_2", "CallType", "call_date")
        df_merged = df_process.join(df1_sel, on="callid", how="left")
        # Define filter values
        level2_filter_values = ["Make Payment", "Missing Payment"]
        df_main = df_merged.filter(col("Level_2").isin(level2_filter_values))
        df_rest = df_merged.filter(~col("Level_2").isin(level2_filter_values))

        # LLM extraction
        result_df = df_main.withColumn(
            "result",
            expr(f"ai_query('{llama_8b}', request => concat({system_prompt_payments_json}, transcript), responseFormat => '{response_format_payment}')")
        )
        response_schema = StructType([
            StructField("payment_related", BooleanType(), True),
            StructField("payment_type", StringType(), True),
            StructField("payment_method", StringType(), True),
            StructField("payment_timeline_provided", BooleanType(), True),
            StructField("payment_timeline_text", StringType(), True)
        ])
        df_parsed = result_df.withColumn("response_parsed", from_json(col("result"), response_schema))
        # df_parsed.toPandas().to_excel('../temp/payment_timescale.xlsx')
        df_final = df_parsed.select(
            "*",
            col("response_parsed.payment_related").alias("payment_related"),
            col("response_parsed.payment_type").alias("payment_type"),
            col("response_parsed.payment_method").alias("payment_method"),
            col("response_parsed.payment_timeline_provided").alias("payment_timeline_provided"),
            col("response_parsed.payment_timeline_text").alias("payment_timeline_text")
        ).drop("response_parsed", "result")

        # UDF for scoring
        def _add_working_days(start_date, days):
            dt = datetime.strptime(start_date, "%d/%m/%Y") if "/" in start_date else datetime.strptime(start_date, "%Y-%m-%d")
            added = 0
            while added < days:
                dt += timedelta(days=1)
                if dt.weekday() < 5:
                    added += 1
            return dt

        def _parse_payment_date(text, ref_date_str):
            return _parse_delivery_date(text, ref_date_str)
        
        def _get_intent(text):
            if not isinstance(text, str):
                return "Others"
            text = text.lower()
            if (('secure' in text and ' line' in text) or \
                ('enter' in text and 'card number' in text) or \
                ('enter' in text and ('security number' in text or 'security code' in text))):
                return "Make a payment"
            else:
                return "Others"

        # def _score(payment_type, payment_method, call_date, payment_timeline_provided, payment_timeline_text,rule_eng_intent,provided_correct_timeline):
        #     try:
        #         if (not payment_type or payment_type == "") or (not payment_method or payment_method == "") or not call_date:
        #             return 10
        #         if rule_eng_intent != "Make a payment":
        #             return 10
        #         if str(payment_timeline_provided).lower()=="false":
        #             return 1
        #         parsed_date = _parse_payment_date(col("payment_timeline_text"), date_format(col("call_date"), "dd/MM/yyyy"))
        #         if not parsed_date:
        #             return 10
        #         if payment_type == "make_payment" and payment_method.lower() == "card payment":
        #             target = _add_working_days(call_date, 1)
        #             return 1 if parsed_date <= target else 0
        #         if payment_type == "missing_payment" and payment_method.lower() in ["bank transfer", "internet banking"]:
        #             target_3 = _add_working_days(call_date, 3)
        #             target_5 = _add_working_days(call_date, 5)
        #             if parsed_date < target_3:
        #                 return 0
        #             elif parsed_date > target_5:
        #                 return 1
        #             else:
        #                 return 10
        #         return 10
        #     except Exception:
        #         return 10
        def _score(payment_type, payment_method, call_date, payment_timeline_provided, payment_timeline_text,rule_eng_intent,provided_correct_timeline,parsed_timeline):
            try:
                if (not payment_type or payment_type == "") or (not payment_method or payment_method == "") or not call_date:
                    return 10, ""
                if rule_eng_intent != "Make a payment":
                    return 10, ""
                if str(payment_timeline_provided).lower()=="false":
                    return 1, "Agent did not provide payment timescale"

                if not parsed_timeline:
                    return 10, ""
            
                if payment_type == "make_payment" and payment_method.lower() == "card payment":
                    if provided_correct_timeline==False:
                        return 1, "Agent provided incorrect payment timescale"
                    else:
                        return 10, ""
                # if payment_type == "missing_payment" and payment_method.lower() in ["bank transfer", "internet banking"]:
                #     target_3 = _add_working_days(call_date, 3)
                #     target_5 = _add_working_days(call_date, 5)
                #     if parsed_date < target_3:
                #         return 0
                #     elif parsed_date > target_5:
                #         return 1
                #     else:
                #         return 10
                return 10, ""
            except Exception as e:
                print(e)
                return 10, ""
    

        # from pyspark.sql.types import StructType, StructField, IntegerType, StringType

        score_schema = StructType([
            StructField("score_payment", IntegerType(), True),
            StructField("comment_payment", StringType(), True)
        ])

        score_udf = udf(_score, score_schema)

        # score_udf = udf(_score, IntegerType())

        df_final = df_final.withColumn(
            "rule_eng_intent",
            udf(_get_intent, StringType())(col("transcript"))
        )

        df_final = add_payment_expected_date_features(df_final)

        # df_final = df_final.withColumn(
        #     "score_payment",
        #     score_udf(
        #         col("payment_type"),
        #         col("payment_method"),
        #         col("call_date"),
        #         col("payment_timeline_provided"),
        #         col("payment_timeline_text"),
        #         col("rule_eng_intent"),
        #         col("provided_correct_timeline")
        #     )
        # )

        df_final = df_final.withColumn("score_payment", score_udf(
            col("payment_type"),
            col("payment_method"),
            col("call_date"),
            col("payment_timeline_provided"),
            col("payment_timeline_text"),
            col("rule_eng_intent"),
            col("provided_correct_timeline"),
            col("timeline_provided")
        ).getField("score_payment"))

        df_final = df_final.withColumn("payment_comments", score_udf(
            col("payment_type"),
            col("payment_method"),
            col("call_date"),
            col("payment_timeline_provided"),
            col("payment_timeline_text"),
            col("rule_eng_intent"),
            col("provided_correct_timeline"),
            col("timeline_provided")
        ).getField("comment_payment"))

        df_final = df_final.withColumn(
            "result_payment",
            when(col("score_payment") == 10, lit("Compliant - Good Outcome"))
            .when(col("payment_comments").contains("Agent did not provide payment timescale"), lit("Compliant with Development"))
            .when(col("payment_comments").contains("Agent provided incorrect payment timescale"), lit("Compliant with Development"))
            .otherwise(lit(None))
        )

        # For rest, set all fields to None/blank and score=10
        df_rest_aug = df_rest.withColumn("score_payment", lit(10))
        missing_cols = [c for c in df_final.columns if c not in df_rest_aug.columns]
        for col_name in missing_cols:
            df_rest_aug = df_rest_aug.withColumn(col_name, lit(None))

        df_concat = df_final.unionByName(df_rest_aug)
        df_concat.toPandas().to_excel('./intermediate_payment_timescale.xlsx')
        return df_concat
    except PySparkException as ex:
        if ex.getErrorClass() == "TABLE_OR_VIEW_NOT_FOUND":
            logger.error("'Payment-Module': 'payment_process': Table or view not found: " + ex.getMessageParameters()['relationName'])
        else:
            logger.error("'Payment-Module':'payment_process': PySparkException in payment_process: ", ex)
        raise
    except Exception as e:
        logger.error("'Payment-Module':Error in payment_process function: ", e)
        raise

def cancellation_process(df_process, df1_process, logger):
    """
    Cancellation Processing via Logic and LLM.
    - Filter by description: 'PARCEL RETURNED UNCOLLECTED' and 'Stop and Return'.
    - For 'PARCEL RETURNED UNCOLLECTED': refund timeline <= 7 working days: score 0, >7: score 1, else 10.
    - For 'Stop and Return': refund timeline <= 3 working days: score 0, >3: score 1, else 10.
    - If refund date falls on Sat/Sun, treat as Tuesday.
    - All other rows: score 10.
    """
    try:
        system_prompt_cancellation_json = json.dumps(system_prompt_cancellation_refund_timeline)
        df_process = df_process.withColumn("transcript", col("transcript").cast("string"))
        df1_sel = df1_process.select("callid", "description", "call_date")
        df_merged = df_process.join(df1_sel, on="callid", how="left")

        filter_values = ["PARCEL RETURNED UNCOLLECTED", "Stop and Return"]
        df_main = df_merged.filter(col("description").isin(filter_values))
        df_rest = df_merged.filter(~col("description").isin(filter_values))

        result_df = df_main.withColumn(
            "result",
            expr(f"ai_query('{llama_8b}', request => concat({system_prompt_cancellation_json}, transcript), responseFormat => '{response_format_cancellation}')")
        )

        response_schema = StructType([
            StructField("cancellation_requested", BooleanType(), True),
            StructField("refund_timeline_provided", BooleanType(), True),
            StructField("refund_timeline_text", StringType(), True)
        ])

        df_parsed = result_df.withColumn("response_parsed", from_json(col("result"), response_schema))
        df_final = df_parsed.select(
            "*",
            col("response_parsed.cancellation_requested").alias("cancellation_requested"),
            col("response_parsed.refund_timeline_provided").alias("refund_timeline_provided"),
            col("response_parsed.refund_timeline_text").alias("refund_timeline_text")
        ).drop("response_parsed", "result")

        def _add_working_days(start_date, days):
            dt = datetime.strptime(start_date, "%d/%m/%Y") if "/" in start_date else datetime.strptime(start_date, "%Y-%m-%d")
            added = 0
            while added < days:
                dt += timedelta(days=1)
                if dt.weekday() < 5:
                    added += 1
            return dt

        def _parse_refund_date(text, ref_date_str):
            return _parse_delivery_date(text, ref_date_str)

        def _adjust_for_weekend(dt):
            if not dt:
                return None
            if dt.weekday() == 5:  # Saturday
                return dt + timedelta(days=3)
            if dt.weekday() == 6:  # Sunday
                return dt + timedelta(days=2)
            return dt

        def _score_cancellation(description, call_date, refund_timeline_text):
            try:
                if not description or not call_date or not refund_timeline_text:
                    return 10
                parsed_date = _parse_refund_date(refund_timeline_text, call_date)
                if not parsed_date:
                    return 10
                parsed_date = _adjust_for_weekend(parsed_date)
                call_dt = datetime.strptime(call_date, "%d/%m/%Y") if "/" in call_date else datetime.strptime(call_date, "%Y-%m-%d")
                if description == "PARCEL RETURNED UNCOLLECTED":
                    target = _add_working_days(call_date, 7)
                    if parsed_date < target:
                        return 0
                    elif parsed_date > target:
                        return 1
                    else:
                        return 10
                elif description == "Stop and Return":
                    target = _add_working_days(call_date, 3)
                    if parsed_date < target:
                        return 0
                    elif parsed_date > target:
                        return 1
                    else:
                        return 10
                else:
                    return 10
            except Exception:
                return 10

        score_udf = udf(_score_cancellation, IntegerType())

        df_final = df_final.withColumn(
            "score_cancellation",
            score_udf(
                col("description"),
                col("call_date"),
                col("refund_timeline_text")
            )
        )

        df_rest_aug = df_rest.withColumn("cancellation_requested", lit(None).cast(BooleanType())) \
            .withColumn("refund_timeline_provided", lit(None).cast(BooleanType())) \
            .withColumn("refund_timeline_text", lit(None).cast(StringType())) \
            .withColumn("score_cancellation", lit(10).cast(IntegerType()))

        df_concat = df_final.unionByName(df_rest_aug)
        return df_concat
    except PySparkException as ex:
        if ex.getErrorClass() == "TABLE_OR_VIEW_NOT_FOUND":
            logger.error("'Cancellation-Module': 'cancellation_process': Table or view not found: " + ex.getMessageParameters()['relationName'])
        else:
            logger.error("'Cancellation-Module':'cancellation_process': PySparkException in cancellation_process: ", ex)
        raise
    except Exception as e:
        logger.error("'Cancellation-Module':Error in cancellation_process function: ", e)
        raise

    

def merge_timescales_and_delivery(df_process, df1_process, logger):
    """
    Calls timescales_process, delivery_process, payment_process, and cancellation_process,
    merges the resulting DataFrames, and drops duplicate columns, keeping only one copy of each.
    """
    df_timescales = timescales_process(df_process, df1_process, logger)  # refund timescales
    df_timescales = validate_refund_markdowns(df_timescales)
    #df_delivery = delivery_process(df_process, df1_process, logger)
    df_payment = payment_process(df_process, df1_process, logger)
    #df_cancellation = cancellation_process(df_process, df1_process, logger)

    join_key = "callid"

    #delivery_unique_cols = [c for c in df_delivery.columns if c not in df_timescales.columns or c == join_key]
    payment_unique_cols = [c for c in df_payment.columns if c not in df_timescales.columns or c == join_key]
    #cancellation_unique_cols = [c for c in df_cancellation.columns if c not in df_timescales.columns and c not in df_delivery.columns and c not in df_payment.columns or c == join_key]

    merged_df = df_timescales.join(
        df_payment.select(*payment_unique_cols),
        on=join_key,
        how="outer"
    )
    # ).join(
    #     df_cancellation.select(*cancellation_unique_cols),
    #     on=join_key,
    #     how="outer"
    # )
    merged_df = merged_df.withColumn("score_cancellation", lit(10))
    merged_df = merged_df.withColumn("score_delivery", lit(10))
    df_comments = comments_results(merged_df, logger)
    df_comments = df_comments.withColumn(
        "score_final_timescales",
        when(col("results_timescales") == "Compliant - Good Outcome", lit(10)).otherwise(col("score_final_timescales"))
    )
    df_sorted = df_comments.orderBy(col("pk").desc())
    df_deduped = df_sorted.dropDuplicates(["pk"])

    return df_deduped

req_level2 = ["Delivery Charge", "Collection Charge","INR - Courier Not Billed",
            "INR - Courier Billed",
            "INR - Store Billed",
            "INR - Store Not Billed",
            "INR - Upholstery - Billed",
            "INR - Upholstery Not Billed",
            "INR - Wrong Address/Store Billed",
            "INR - Wrong Address/Store Not Billed",
            "Delivery Charge",
            "Collection Charge",
            "Returns Not Credited - Store ",
            "Returns Not Credited - Courier",
            "Refund - Not Received",
            "Not In Parcel Billed",
            "Not in Parcel Not Billed",
            "Returns not credited - Supplier ",
            "INR - Parcel shop Billed",
            "INR - Parcel Shop Not Billed",
            "Returns Not Credited - Carrier",
            "Returns Not Credited - Evri Locker",
            "Returns Not Credited - Parcelshop",
            "Returns Not Credited - Post",
            "Returns Not Credited - Store ",
            "Refund - Not Received",

            "Refund - Not Received",
            "Returns Not Credited - Carrier",
            "Returns Not Credited - Courier",
            "Returns Not Credited - Evri Locker",
            "Returns Not Credited - Parcelshop",
            "Returns Not Credited - Post",
            "Returns Not Credited - Store",
            "Returns Not Credited - Supplier",
            "GNR - 3rd Party Supplier - Billed",
            "GNR - 3rd Party Supplier Not Billed",
            "GNR - Courier Billed",
            "GNR - Courier Not Billed",
            "GNR - Failed Delivery Billed",
            "GNR - Failed Delivery Not Billed",
            "GNR - International Carrier",
            "GNR - Parcel shop Billed",
            "GNR - Parcel Shop Not Billed",
            "GNR - Store Billed",
            "GNR - Store Not Billed",
            "GNR - Upholstery - Billed",
            "GNR - Upholstery Not Billed",
            "GNR - Wrong Address/Store Billed",
            "GNR - Wrong Address/Store Not Billed",
            "Not In Parcel Billed",
            "Not in Parcel Not Billed",
            "Interest charge",
            "Make Payment", "Missing Payment"
            ]

def calculate_final_result(row):
    priority_dict = {
    "Compliant - Good Outcome": 3,  #Lowest
    "Compliant with Development": 2,
    "No Poor outcome": 1
    }

    
    payment_priority = priority_dict.get(row.get('result_payment'), 404)
    refund_priority = priority_dict.get(row.get('results_refund'), 404)

    priority = min(payment_priority, refund_priority)
    if priority == 404:
        print("Missing result..")
        priority = priority_dict["Compliant - Good Outcome"]

    reverse_priority_dict = {v: k for k, v in priority_dict.items()}
    score_map_dict = {
        "No Poor outcome": 1,
        "Compliant with Development": 1,
        "Compliant - Good Outcome": 10
    }
    row['results_timescales'] = reverse_priority_dict[priority]
    row['score_final_timescales'] = score_map_dict[row['results_timescales']]
    # Add comment of parameter which gave score 0
    if priority == payment_priority and row['score_payment'] in [0,1]:
        row['comments_timescales'] = row['payment_comments']
    elif priority == refund_priority and row['score_refund'] in [0,1]:
        row['comments_timescales'] = row['comments_refund']
    else:
        row['comments_timescales'] = ""
    if row['Level_2'] not in req_level2:
        row['score_final_timescales'] = 10
        row['results_timescales'] = "N.A"
        row['comments_timescales'] = ""
        return row
    return row


if __name__=="__main__":
    logger = get_logger()
    try:
        logger.info("Starting the Timescales-Module!!")
        main_config = yaml.safe_load(open('../main_config.yaml', 'r'))
        llama_8b = main_config['LLM']['llama']
        catalog_config = yaml.safe_load(open('../catalog_config.yaml', 'r'))
        # try:
        #     formatted_date = dbutils.jobs.taskValues.get(taskKey="Pre-Modules-Run", key="sql_date")
        # except Exception as e:
        #     formatted_date = sys.argv[1]
        #     if not re.match(r"^\d{4}-\d{2}-\d{2}$", str(formatted_date)):
        #         raise ValueError(f"formatted_date '{formatted_date}' is not in yyyy-mm-dd format")
        #     logger.info(f"formatted_date: {formatted_date}")
        formatted_date = "1999-01-01"
        call_ids = [
            "ed33a20d-20d8-40da-ad1d-a0b2a69dcad1",
            "63581ae9-ba1f-4d67-85ea-5ecb8b7090d9",
            "79ee4a85-e424-4b66-88b9-c7ca7e3f3cd2",
            "0dbdd5b6-1c39-4731-84ec-0b13f53a08af",
            "2f8850be-9542-428b-8044-8f6d53bad09b",
            "456b5063-e9d0-4699-88a6-7966bf882080",
            "61a1d404-e6da-47ab-b3ba-fe9809dabb39",
            "adeb1020-6ba9-4556-b7b3-d2ea4c26b8e8",
            "0d93a184-b01d-4720-ae39-a5edba352d0f",
            "08022d37-d4e9-4c85-b39b-1af06b7a4023",
            "33642d7c-3e61-4798-84d3-fd2ce05263e3",
            "46ccb943-7388-4e35-a308-274879768dea",
            "a62b41ea-fa66-4d65-b1af-29f2596778db",
            "bcc6e504-48e7-47fe-b9c7-7f8c7de2dca5",
            "f4089215-96c2-4821-ba37-5f1dd81aef15",
            "cbc254a4-0afb-4599-9853-f9bbb81cf001"
        ]
        call_ids_str = ",".join([f"'{cid}'" for cid in call_ids])
        INPUT_FILE_PATH = spark.sql(f"""
            SELECT *
            FROM contactcentre_prod.iaudit.filtered_input_data_v2_iaudit
            WHERE callid IN ({call_ids_str})
        """)
        # INPUT_FILE_PATH = spark.table(catalog_config['output_table']['filtered_table']).filter(f"call_date = '{formatted_date}'")

        df =  INPUT_FILE_PATH.toPandas()
        # df = df[df['callid'].isin([])]

        # OUTPUT_FILE = f"./Output/qca/timescales_module_output_{formatted_date}.xlsx"
        # os.makedirs(os.path.dirname(OUTPUT_FILE), exist_ok=True)
        # df = pd.read_excel(INPUT_FILE_PATH)

        grouped_df = group(df, logger)
        print(grouped_df.columns)
        spark_df = spark.createDataFrame(grouped_df)
        #sample_len = spark_df.count()//2
        # temp_df = spark_df.orderBy(col("callid").desc()).limit(20)
        temp_df = spark_df
        
        callids = [row['callid'] for row in temp_df.select("callid").distinct().collect()]
        display(spark.createDataFrame([{"callids_length": len(callids)}]))
        callids_str = ",".join([f"'{cid}'" for cid in callids])
        query = f"""
            SELECT * FROM contactcentre_prod.staging.zen_live9
            WHERE conversation_id IN ({callids_str})
        """
        df_spark = spark.sql(query)
        # display(spark.createDataFrame([{"unique_callids_count": df_spark.select("conversation_id").distinct().count()}]))


        df_spark = df_spark.withColumnRenamed("ResponseName", "Level_2")
        df_spark = df_spark.withColumnRenamed("conversation_id", "callid")
        df_spark = df_spark.withColumnRenamed("account_number", "account_no")
        df_spark = df_spark.select("callid", "call_date", "Level_2" , "account_no","account_type","CallType")

        res = merge_timescales_and_delivery(temp_df, df_spark, logger)
        # res.write.format("delta").mode("overwrite").saveAsTable(catalog_config['intermediate']['escalation'])
    
        # now = datetime.now()
        # date_string = now.strftime("%Y_%m_%d")


        pandas_df = res.toPandas()
        
        pandas_df = pandas_df.apply(calculate_final_result, axis=1)
        
        # for col in ['Score']:
        #     pandas_df[col] = pd.to_numeric(pandas_df[col], errors='coerce')
        pandas_df.columns = [col.replace(' ', '_') for col in pandas_df.columns]
        pandas_df['call_date'] = pd.to_datetime(formatted_date)
        pandas_df.to_excel(f'./timescales_{formatted_date}.xlsx')
        final_res = spark.createDataFrame(pandas_df)

        final_res = final_res.withColumn("score_payment", col("score_payment").cast("double"))

        ## CAST
        # Schema dictionary map: column_name -> target_type
        schema_map = {
            "callid":                     StringType(),
            "pk":                         StringType(),
            "transcript":                 StringType(),
            "Level_2":                    StringType(),
            "CallType":                   StringType(),
            "call_date":                  TimestampType(),
            "account_no":                 StringType(),
            "refund_or_adjustment_made":  StringType(),
            "type_of_adjustment":         StringType(),
            "refund_timeline_provided":   StringType(),
            "refund_timeline_text":       StringType(),
            "score_refund":               DoubleType(),
            "exp_delv_date":              DateType(),
            "delivery_promised":          BooleanType(),
            "delivery_date_provided":     BooleanType(),
            "delivery_date_text":         StringType(),
            "extracted_date":             DateType(),
            "score_delivery":             IntegerType(),
            "payment_related":            BooleanType(),
            "payment_type":               StringType(),
            "payment_method":             StringType(),
            "payment_timeline_provided":  BooleanType(),
            "payment_timeline_text":      StringType(),
            "score_payment":              DoubleType(),
            "score_cancellation":         IntegerType(),
            "score_final_timescales":     IntegerType(),
            "comments_timescales":        StringType(),
            "results_timescales":         StringType(),
            "refund_val_LLM":             StringType(),
            "positive_one_day":           BooleanType(),
            "rule_eng_intent":            StringType(),
            "createdate":                 TimestampType(),
            "dow":                        StringType(),
            "hour":                       DoubleType(),
            "is_after_10pm":              BooleanType(),
            "timeline_provided":          DateType(),
            "calculated_expected_date":   DateType(),
            "provided_correct_timeline":  BooleanType(),
            "payment_comments":           StringType(),
            "comments_refund":            StringType(),
            "results_refund":             StringType(),
            "result_payment":             StringType(),
        }

        # Cast all columns in df to the target schema
        def cast_to_schema(df, schema_map):
            for col_name, col_type in schema_map.items():
                if col_name in df.columns:
                    df = df.withColumn(col_name, F.col(col_name).cast(col_type))
            return df

        # Usage
        final_res = cast_to_schema(final_res, schema_map)     
        final_res = final_res.select([col for col in schema_map.keys() if col in final_res.columns])  # Remove columns not in schema.


        # Add final output to DB 
        filtered_table = catalog_config['intermediate']['timescales']
        date_exists = (
            spark.table(filtered_table)
            .filter(F.col("call_date") == formatted_date)
            .limit(1)
            .count() > 0
        )
        if date_exists:
            logger.info(f"Results for {formatted_date} already exists. Deleting old data before insert.")
            spark.sql(f"DELETE FROM {filtered_table} WHERE call_date = '{formatted_date}'") 

        final_res.write.format("delta").option("mergeSchema", "true").mode("append").saveAsTable(catalog_config['intermediate']['timescales'])
        #Data pushed to DB

        # pandas_df.to_excel(OUTPUT_FILE, index=False)
        # logger.info(f"Timescales-Module completed and save resullts to a catalog: {catalog_config['intermediate']['escalation']}")
        logger.info(f"Timescales-Module completed and save resullts to a catalog:")
        
        #Garbage Collection
        
        try:
            del df
            del grouped_df
            del spark_df
            del temp_df
            del callids
            del callids_str
            del df_spark
            del res
            del pandas_df
        except:
            pass

    except FileNotFoundError as e:
        logger.error("'Timescales-Module': Input file not found: %s", e)
    except PySparkException as ex:
        if ex.getErrorClass() == "TABLE_OR_VIEW_NOT_FOUND":
            logger.error("'Timescales-Module':'Table or view not found: " + ex.getMessageParameters()['relationName'])
        else:
            logger.error("'Timescales-Module':PySparkException in main: %s", ex)
        raise
    except Exception as e:
        logger.error("'Timescales-Module':Unhandled error in main: %s", e)
        raise