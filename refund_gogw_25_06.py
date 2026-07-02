# !pip install json-repair==0.58.5
# import importlib
# import prompts.refund_gogw_prompts
# importlib.reload(refund_gogw_prompts)
# !pip install openai
import pandas as pd
import numpy as np
from openai import OpenAI, RateLimitError
import os, sys, yaml
import time
import json
from json_repair import repair_json
from pyspark.sql.functions import expr
from pyspark.sql.functions import col
from prompts.refund_gogw_prompts import (
    prompt_proof_verification,
    prompt_price_mismatch,
    refund_promise_prompt,
    prompt_item_replaced,
    prompt_refund_delivery_collection,
    prompt_gogw_check,
    prompt_justified_refund_reason,
    prompt_refund_amount_reconfirmation
)
import re
sys.path.append('../')
from utils.iaudit_logger import get_logger
from pyspark.sql.types import *
import pyspark.sql.functions as F
logger = get_logger()

# How to get your Databricks token: https://docs.databricks.com/en/dev-tools/auth/pat.html
#DATABRICKS_TOKEN = os.environ.get('DATABRICKS_TOKEN')
# Alternatively in a Databricks notebook you can use this:
DATABRICKS_TOKEN = dbutils.notebook.entry_point.getDbutils().notebook().getContext().apiToken().get()

client = OpenAI(
    api_key=DATABRICKS_TOKEN,
    base_url="https://adb-6188831950334199.19.azuredatabricks.net/serving-endpoints"
)

main_config = yaml.safe_load(open('../main_config.yaml', 'r'))
catalog_config = yaml.safe_load(open('../catalog_config.yaml', 'r'))

llama_8b = main_config['LLM']['llama']
llama_70b = main_config['LLM']['llama_large']

# model_small="databricks-meta-llama-3-1-8b-instruct"
# model_small_provisioned = "contact_centre_internal_batch"
# model_large = "databricks-meta-llama-3-3-70b-instruct"
model = llama_70b

# model_large = "databricks-qwen3-next-80b-a3b-instruct",

def call_llm(prompt, max_retries=12, initial_wait=5):
    # logger.info(f"Calling LLM with prompt: {prompt[:500]}...")  # Log first 500 chars for brevity
    # prompt = prompt.replace("'", "\\\\'")
    for attempt in range(1, max_retries + 1):
        try:
            response = client.chat.completions.create(
                model = model,
                messages=[
                    {
                        "role": "user",
                        "content": prompt
                    }
                ],
                temperature=0.0,
                top_p=1.0,
                max_tokens=512
            )
            # logger.info(f"LLM response: {response.choices[0].message.content[:500]}...")  # Log first 500 chars for brevity
            return response.choices[0].message.content
        except RateLimitError as e:
            wait_time = initial_wait * (2 ** (attempt - 1))  # exponential backoff: 5, 10, 20, 40, ...
            logger.warning(f"Rate limit hit (attempt {attempt}/{max_retries}). Waiting {wait_time}s before retry...")
            print(f"Rate limit hit (attempt {attempt}/{max_retries}). Waiting {wait_time}s before retry...")
            if attempt == max_retries:
                logger.error(f"Max retries ({max_retries}) exceeded. Raising RateLimitError.")
                raise
            time.sleep(wait_time)
    return ""

# def get_llm_response(transcript, prompt):
#     full_prompt_refund = prompt + "\n\nTranscript:\n" + transcript
#     # logger.info(f"Generating LLM response for transcript (call_llm input length: {len(full_prompt_refund)})")
#     llm_response = call_llm(full_prompt_refund)
#     # logger.info(f"LLM response for transcript: {llm_response[:500]}...")
#     return llm_response

def call_llm_spark(model, prompt, df_spark, transcript_col, output_column):
    logger.info(f"Starting LLM call with model: {model}, output_column: {output_column}, rows: {df_spark.count()}")
    params = {"temperature":0.1,"top_p":0.95,"max_new_tokens":800}
    # model = "contact_centre_internal_batch_large"

    # prompt = prompt.replace("'", "\\\\'")
    prompt = prompt.replace("'", "''")
    ai_query_expr =f"ai_query('{model}',request => concat('Instructions: ','{prompt}', '\n', '\n\nTranscript:\n',{transcript_col}), modelParameters => named_struct('temperature', {params.get('temperature',0.1)}, 'top_p', {params.get('top_p',0.95)}))"
    
    result_df = df_spark.withColumn(
        output_column,
        expr(ai_query_expr)
        )
    logger.info(f"Completed LLM call for {output_column}")
    return result_df

def preprocess_transcript(df):    
    logger.info("Preprocessing transcript DataFrame")
    df['channel_transcript'] = df['channel'].astype(str) + ": " + df['transcript'].astype(str)
    merged_df = df.groupby(['callid', 'AgentID']).apply(
        lambda x: pd.Series({
            'channel_transcript': "\n".join(x['channel_transcript']),
            'createdate': x['createdate'].iloc[0],
            'queue': x['queue'].iloc[0]
        })
    ).reset_index()
    merged_df['pk'] = merged_df['callid'].astype(str) + "|" + merged_df['AgentID'].astype(str)
    logger.info(f"Preprocessed transcript shape: {merged_df.shape}")
    return merged_df

def split_last_prev_callers(merged_df):
    logger.info("Splitting last and previous callers")
    # Convert createdate to datetime if not already
    merged_df['createdate'] = pd.to_datetime(merged_df['createdate'])

    # Find callids with more than 2 occurrences
    callid_counts = merged_df['callid'].value_counts()
    callids_gt2 = callid_counts[callid_counts >= 2].index

    # Split the DataFrame
    recent_rows = []
    prev_rows = []

    for callid in callids_gt2:
        subset = merged_df[merged_df['callid'] == callid]
        subset_sorted = subset.sort_values('createdate', ascending=False)
        recent = subset_sorted.iloc[[0]]
        prev = subset_sorted.iloc[1:]
        recent_rows.append(recent)
        prev_rows.append(prev)

    # DataFrames for recent and previous occurrences
    df_recent = pd.concat(recent_rows) if recent_rows else pd.DataFrame(columns=merged_df.columns)
    df_prev_caller = pd.concat(prev_rows) if prev_rows else pd.DataFrame(columns=merged_df.columns)

    # For callids with 2 or fewer occurrences, keep as is in merged_df
    callids_le2 = callid_counts[callid_counts < 2].index
    df_le2 = merged_df[merged_df['callid'].isin(callids_le2)]

    # Final merged_df: only recent for callids >2, plus all for callids <=2
    merged_df = pd.concat([df_le2, df_recent], ignore_index=True)
    logger.info(f"Split complete: prev_caller shape {df_prev_caller.shape}, merged_df shape {merged_df.shape}")
    return df_prev_caller, merged_df


def extract_mismatch_info(transcript, prompt):
    full_prompt = prompt + "\n\nTranscript:\n" + transcript
    # logger.info("Extracting mismatch info from transcript")
    llm_response = call_llm(full_prompt)
    try:
        llm_response1 = repair_json(llm_response)
        response_dict = json.loads(llm_response1)
        mismatch_issue = response_dict.get("Price mismatch issue", "")
        type_of_mismatch = response_dict.get("Type of mismatch", "")
        evidence = response_dict.get("Evidence", "")
        # logger.info(f"Extracted mismatch info: {response_dict}")
        # print(llm_response)
    except Exception:
        logger.error(f"Error parsing JSON response: {llm_response}")
        print(f"Error parsing JSON response: {llm_response}")
        mismatch_issue = ""
        type_of_mismatch = ""
        evidence = ""
    return pd.Series([mismatch_issue, type_of_mismatch, evidence, llm_response])

def get_logger_data(subset_df1):
    logger.info(f"Fetching logger data for {len(subset_df1)} callids")
    callids = subset_df1['callid'].tolist()
    callids_str = ",".join([f"'{c}'" for c in callids]) 
    logger_query = f"""
        SELECT conversation_id, account_number, ResponseName, call_date
        FROM contactcentre_prod.staging.zen_live9
        WHERE conversation_id IN ({callids_str})
        """
    logger.info(f"Running logger query..")
    logger_table = spark.sql(logger_query)
    logger_table_pandas = logger_table.toPandas()
    logger.info(f"Fetched {len(logger_table_pandas)} logger records")
    return logger_table_pandas

def read_adjustment_data(adjustment_path, adjustment_sheet, gift_card_sheet):
    logger.info(f"Reading adjustment data from {adjustment_sheet}")
    df_adjustment =pd.read_excel(adjustment_path, sheet_name = adjustment_sheet)
    df_giftcard =pd.read_excel(adjustment_path, sheet_name = gift_card_sheet)
    logger.info(f"Adjustment data loaded - Adjustments: {df_adjustment.shape}, Gift cards: {df_giftcard.shape}")

    # Parse the TimeStamp column (format: "dd/MM/yyyy HH:mm")
    df_adjustment["TimeStamp"] = pd.to_datetime(
        df_adjustment["TimeStamp"], format="%d/%m/%Y %H:%M", errors="coerce"
    )
    # df_adjustment["call_date"] = df_adjustment["TimeStamp"].dt.strftime("%Y-%m-%d")
    df_adjustment["call_date"] = df_adjustment["DateStamp"].dt.strftime("%Y-%m-%d")
    df_giftcard["call_date"] = df_giftcard["Date Created"].dt.strftime("%Y-%m-%d")

    adjustment_columns = {'Account Number': 'account_number', 'Account Type': 'account_type_adj', 'Adj Type': 'adj_type', 'Adj Amount': 'adj_amount', 'Adj Reason': 'adj_reason', 'Adj Reason Code': 'adj_reason_code', 'Subcategory': 'subcategory_adj', 'Item Number': 'item_number_adj', 'Item Description': 'item_description_adj', 'Item Size': 'item_size_adj', 'Return Method': 'return_method_adj', 'call_date': 'call_date'}
    df_adjustment = df_adjustment[adjustment_columns.keys()].rename(columns=adjustment_columns)
    df_adjustment = df_adjustment.groupby(['account_number', 'call_date']).agg(lambda x: '\n'.join(x.astype(str))).reset_index()

    giftcards_columns={'Account No': 'account_number', 'Reason': 'reason_giftcard', 'Currency': 'currency_giftcard', 'Value': 'value_giftcard', 'Authorising TM': 'authorising_tm', 'Brand': 'brand', 'Year': 'year', 'Week': 'week', 'Type': 'type_giftcard','call_date': 'call_date'}
    df_giftcard = df_giftcard[giftcards_columns.keys()].rename(columns=giftcards_columns)
    df_giftcard = df_giftcard.groupby(['account_number', 'call_date']).agg(lambda x: '\n'.join(x.astype(str))).reset_index()

    merged_logger['account_number'] = merged_logger['account_number'].astype(str)
    df_adjustment['account_number'] = df_adjustment['account_number'].astype(str)
    df_giftcard['account_number'] = df_giftcard['account_number'].astype(str)

    merged_logger['call_date'] = merged_logger['call_date'].astype(str)
    df_adjustment['call_date'] = df_adjustment['call_date'].astype(str)
    df_giftcard['call_date'] = df_giftcard['call_date'].astype(str)

    merged_adjustment = merged_logger.merge(
        df_adjustment,
        left_on=['account_number', 'call_date'],
        right_on=['account_number', 'call_date'],
        how='left'
    )
    merged_giftcard = merged_adjustment.merge(
        df_giftcard,
        left_on=['account_number', 'call_date'],
        right_on=['account_number', 'call_date'],
        how='left'
    )
    logger.info(f"Merged adjustment and giftcard data shape: {merged_giftcard.shape}")
    return merged_giftcard

def fetch_payin3_accounts(merged_giftcard):
    customer_account_number_list = merged_giftcard['account_number'].unique().tolist()    
    # Handle empty list or format tuple properly for SQL IN clause
    if len(customer_account_number_list) == 0:
        # Create empty dataframe with expected columns
        df_account_details = pd.DataFrame(columns=['account_number', 'AccountType', 'SpecialAccountDescription'])
        logger.info("No accounts to fetch details for")
    else:
        # Format the list properly for SQL IN clause
        if len(customer_account_number_list) == 1:
            account_filter = f"('{customer_account_number_list[0]}')"
        else:
            account_filter = str(tuple(customer_account_number_list))
        
        logger.info(f"Querying account details for {len(customer_account_number_list)} accounts")
        spark_account_details =  spark.sql(f"""
        SELECT `account_number`, `AccountType`, `SpecialAccountDescription`
        FROM `contactcentre_prod`.`iaudit`.`account_details`
        WHERE `account_number` IN {account_filter}  
        """)
        df_account_details = spark_account_details.toPandas()
        df_account_details = df_account_details.groupby('account_number').agg(lambda x: '\n'.join(x.astype(str))).reset_index()
    
    print("Account types fetched: ", df_account_details.shape)
    logger.info(f"Account details fetched: {df_account_details.shape}")

    payin3_queueids = [
        "237a33e8-0652-49fa-ad09-e64eb85042ae",
        "f44ba38a-7360-43df-adf8-bc3df6bf1fa0",
        "eeada463-9938-47e5-abd8-99266c5ff9f1",
        "f34e420a-5042-4319-bd10-7c990852c876",
        "fccc2859-2fd6-4215-9bd7-4d4400e2084a",
        "f31494de-2f03-4044-9457-3c261e8837e6",
        "2da50e24-de09-4de2-a6cd-c2b3f2f434b9",
        "81cbf37f-394f-473b-af10-4c2de07bd3b9"
    ]
    
    # Add payin3_queue column: True if 'queue' in payin3_queueid, else False
    if 'queue' in merged_giftcard.columns:
        merged_giftcard['payin3_queue'] = merged_giftcard['queue'].isin(payin3_queueids)
    else:
        merged_giftcard['payin3_queue'] = False
    logger.info(f"payin3_queue value counts: {merged_giftcard['payin3_queue'].value_counts().to_dict()}")
    
    merged_giftcard = merged_giftcard.merge(
        df_account_details,
        left_on=['account_number'],
        right_on=['account_number'],
        how='left'
    )
    return merged_giftcard

def check_proof(channel_transcript):
    logger.info("Checking proof in transcript")
    llm_proof_check = call_llm(prompt_proof_verification + "\n\nTranscript:\n" + channel_transcript)
    # logger.info(f"Proof check LLM response: {llm_proof_check}")
    if "true" in llm_proof_check.lower():
        return True
    else:
        return False

def price_match_scoring(df_price_mismatch):
    logger.info("Scoring price match calls")
    price_match_calls = df_price_mismatch.copy()
    price_mismatch_score = []
    price_mismatch_remarks = []
    price_mismatch_comments = []
    for i,row in price_match_calls.iterrows():
        amnt1,amnt2 = [float(i) for i in str(row['adj_amount']).split('\n') if i.strip().lower() not in ('none', 'nan', '')], [float(i) for i in str(row['value_giftcard']).split('\n') if i.strip().lower() not in ('none', 'nan', '')]
        if row['mismatch_issue'] == "True" and 'tag mismatch' in row['type_of_mismatch'].lower():
            if (any(x <= 10 for x in amnt1) or any(x <= 10 for x in amnt2)):
                price_mismatch_score.append(10)
                price_mismatch_remarks.append("Compliant - Good Outcome")
                price_mismatch_comments.append("tag mismatch issue <10, confidence-High")
            # elif (all(pd.isnull(x) or x > 10 for x in amnt1) or all(pd.isnull(x) or x > 10 for x in amnt2)):
            # elif (pd.isnull(amnt2) and any(x > 10 for x in amnt1)) or (pd.isnull(amnt1) and any(x > 10 for x in amnt2)):
            elif ((isinstance(amnt2, list) and all(pd.isnull(x) for x in amnt2)) and any(x > 10 for x in amnt1)) or ((isinstance(amnt1, list) and all(pd.isnull(x) for x in amnt1)) and any(x > 10 for x in amnt2)):
                proof_available = check_proof(row['channel_transcript'])
                if proof_available:
                    price_mismatch_score.append(10)
                    price_mismatch_remarks.append("Compliant - Good Outcome")
                    price_mismatch_comments.append("tag mismatch issue, amnt>10,proof checked, confidence-High")
                elif proof_available==False and 'Price Match' in str(row['adj_reason']):
                    price_mismatch_score.append(1)
                    price_mismatch_remarks.append("Non-Compliant No Poor Outcome")
                    price_mismatch_comments.append("tag mismatch issue, amnt>10,no proof checked, adj_reason-price match, confidence-High")
                else:
                    price_mismatch_score.append(1)
                    price_mismatch_remarks.append("Non-Compliant No Poor Outcome")
                    price_mismatch_comments.append("tag mismatch issue, amnt>10,no proof checked, confidence-Low")
            else:
                if "true" in row['response_refund_promissed']:
                    price_mismatch_score.append(0)
                    price_mismatch_remarks.append("Non-Compliant Poor Outcome")
                    price_mismatch_comments.append("refund promissed, not initiated")
                else:
                    price_mismatch_score.append(10)
                    price_mismatch_remarks.append("Compliant - Good Outcome")
                    price_mismatch_comments.append("")
        elif row['mismatch_issue'] == "True" and 'other website' in row['type_of_mismatch'].lower():
            #check we can not do price match
            if (any(x > 0 for x in amnt1) and row['adj_reason'] == 'Price Match') or any(x > 0 for x in amnt2):
                price_mismatch_score.append(1)
                price_mismatch_remarks.append("Non-Compliant No Poor Outcome")
                price_mismatch_comments.append("Low price on other website, adjustment done, confidence-High")
            elif any(x > 0 for x in amnt1) or any(x > 0 for x in amnt2):
                price_mismatch_score.append(1)
                price_mismatch_remarks.append("Non-Compliant No Poor Outcome")
                price_mismatch_comments.append("Low price on other website, adjustment done, confidence-Low")
            else:
                price_mismatch_score.append(10)
                price_mismatch_remarks.append("Compliant - Good Outcome")
                price_mismatch_comments.append("Low price on other website, no adjustment done, confidence-High")
        elif row['mismatch_issue'] == "True" and 'price decreased' in row['type_of_mismatch'].lower():
            if (any(x > 0 for x in amnt1) and row['adj_reason'] == 'Price Match') or any(x > 0 for x in amnt2):
                price_mismatch_score.append(1)
                price_mismatch_remarks.append("Non-Compliant No Poor Outcome")
                price_mismatch_comments.append("Price decreased after order, adjustment done, confidence-High")
            elif any(x > 0 for x in amnt1) or any(x > 0 for x in amnt2):
                price_mismatch_score.append(1)
                price_mismatch_remarks.append("Non-Compliant No Poor Outcome")
                price_mismatch_comments.append("Price decreased after order, adjustment done, confidence-Low")
            else:
                price_mismatch_score.append(10)
                price_mismatch_remarks.append("Compliant - Good Outcome")
                price_mismatch_comments.append("Price decreased after order, no adjustment done, confidence-High")
        elif row['mismatch_issue'] == "True" and 'price increased' in row['type_of_mismatch'].lower():
            llm_out = call_llm(prompt_item_replaced + "\n\nTranscript:\n" + row['channel_transcript'])
            print(llm_out,"-----------------------------------")
            llm_out = repair_json(llm_out)
            try:
                llm_json = json.loads(llm_out)
            except Exception as e:
                logger.error(f"Error parsing JSON in price increased branch: {llm_out}")
                llm_json = {}
            response_list = list(llm_json.values())
            if "true" in response_list[1] and (any(x > 0 for x in amnt1) or any(x > 0 for x in amnt2)):
                price_mismatch_score.append(1)
                price_mismatch_remarks.append("Non-Compliant No Poor Outcome")
                price_mismatch_comments.append("Price increased for reorder, adjustment done before delivery, confidence-Low")
            elif "true" in response_list[1]:
                price_mismatch_score.append(10)
                price_mismatch_remarks.append("Compliant - Good Outcome")
                price_mismatch_comments.append("Price increased for reorder, no adjustment done before delivery, confidence-High")
            elif ("true" in response_list[0] or "true" in response_list[2]) and (any(x > 0 for x in amnt1) or any(x > 0 for x in amnt2)):
                price_mismatch_score.append(10)
                price_mismatch_remarks.append("Compliant - Good Outcome")
                price_mismatch_comments.append("Price increased for reorder, adjustment done after delivery, confidence-High")
            elif ("true" in response_list[0] or "true" in response_list[2]) and (any(x == 0 for x in amnt1) and any(x == 0 for x in amnt2)):
                price_mismatch_score.append(1)
                price_mismatch_remarks.append("Non-Compliant No Poor Outcome")
                price_mismatch_comments.append("Price increased for reorder, adjustment not done after delivery, confidence-Low")
            else:
                price_mismatch_score.append(10)
                price_mismatch_remarks.append("Compliant - Good Outcome")
                price_mismatch_comments.append("confidence-High")
        else:
            price_mismatch_score.append(10)
            price_mismatch_remarks.append("Compliant - Good Outcome")
            price_mismatch_comments.append("confidence-High")
    # if row['callid']=='1a2860a1-cc20-4ce9-bd38-8207910b0fa9':
    #     break
    price_match_calls['price_mismatch_score'] = price_mismatch_score
    price_match_calls['price_mismatch_remarks'] = price_mismatch_remarks
    price_match_calls['price_mismatch_comments'] = price_mismatch_comments
    logger.info("Price match scoring complete")
    return price_match_calls

def process_price_mismatch(df_price_mismatch):
    logger.info("Processing price mismatch")
    price_match_calls = df_price_mismatch.copy()
    if price_match_calls.empty:
        logger.info("No price mismatch calls to process, returning empty DataFrame")
        for col_name in ["price_mismatch_response", "response_refund_promissed", "mismatch_issue",
                         "type_of_mismatch", "evidence", "price_mismatch_score",
                         "price_mismatch_remarks", "price_mismatch_comments"]:
            price_match_calls[col_name] = []
        return price_match_calls
    # price_match_calls[["mismatch_issue", "type_of_mismatch", "evidence"]] = price_match_calls.apply(lambda row: extract_mismatch_info(row["channel_transcript"], prompt_price_mismatch), axis=1)
    # mismatch_issue_list = []
    # type_of_mismatch_list = []
    # evidence_list = []
    # price_mismatch_response = []
    # for i, row in price_match_calls.iterrows():
    #     # mismatch_issue, type_of_mismatch, evidence, llm_res = extract_mismatch_info(row["channel_transcript"], prompt_price_mismatch)
    #     llm_response = row['price_mismatch_response']
    #     llm_response1 = repair_json(llm_response)
    #     response_dict = json.loads(llm_response1)
    #     mismatch_issue = response_dict.get("Price mismatch issue", "")
    #     type_of_mismatch = response_dict.get("Type of mismatch", "")
    #     evidence = response_dict.get("Evidence", "")

    #     mismatch_issue_list.append(mismatch_issue)
    #     type_of_mismatch_list.append(type_of_mismatch)
    #     evidence_list.append(evidence)
    #     price_mismatch_response.append(llm_res)
    # price_match_calls["mismatch_issue"] = mismatch_issue_list
    # price_match_calls["type_of_mismatch"] = type_of_mismatch_list
    # price_match_calls["evidence"] = evidence_list
    # price_match_calls["price_mismatch_response"] = price_mismatch_response

    # price_match_calls["response_refund_promissed"] = price_match_calls["channel_transcript"].apply(lambda x: get_llm_response(x, refund_promise_prompt))
    df_spark = spark.createDataFrame(price_match_calls)
    df_spark = call_llm_spark(model, prompt_price_mismatch, df_spark, 'channel_transcript', 'price_mismatch_response')

    df_spark = call_llm_spark(model, refund_promise_prompt, df_spark, 'channel_transcript', 'response_refund_promissed')
    price_match_calls = df_spark.toPandas()

    mismatch_issue_list = []
    type_of_mismatch_list = []
    evidence_list = []
    # price_mismatch_response = []
    for i, row in price_match_calls.iterrows():
        try:
            llm_response = row['price_mismatch_response']
            llm_response1 = repair_json(llm_response)
            response_dict = json.loads(llm_response1)
            mismatch_issue = response_dict.get("Price mismatch issue", "")
            type_of_mismatch = response_dict.get("Type of mismatch", "")
            evidence = response_dict.get("Evidence", "")
        except Exception:
            logger.error(f"Error parsing JSON response: {llm_response}")
            print(f"Error parsing JSON response: {llm_response}")
            mismatch_issue = ""
            type_of_mismatch = ""
            evidence = ""
        mismatch_issue_list.append(mismatch_issue)
        type_of_mismatch_list.append(type_of_mismatch)
        evidence_list.append(evidence)
    price_match_calls["mismatch_issue"] = mismatch_issue_list
    price_match_calls["type_of_mismatch"] = type_of_mismatch_list
    price_match_calls["evidence"] = evidence_list

    price_match_calls = price_match_scoring(price_match_calls)
    logger.info("Price mismatch processing complete")
    return price_match_calls

def check_refund_delivery_collection_response(refund_collections_df):
    logger.info("Checking refund delivery/collection responses")
    if refund_collections_df.empty:
        refund_collections_df['refund_delivery_collection_response'] = []
        return refund_collections_df
    # refund_delivery_collection_responses = []
    # for i, row in refund_collections_df.iterrows():
    #     # logger.info(f"Processing callid: {row['callid']}")
    #     channel_transcript = row['channel_transcript']
    #     response = call_llm(prompt_refund_delivery_collection + "\n\nTranscript:\n" + channel_transcript)
    #     # logger.info(f"Refund delivery/collection LLM response: {response}")
    #     # print(response)
    #     print("-----------------------")
    #     refund_delivery_collection_responses.append(response)
    # #
    # refund_collections_df['refund_delivery_collection_response'] = refund_delivery_collection_responses

    df_spark = spark.createDataFrame(refund_collections_df)
    df_spark = call_llm_spark(model, prompt_refund_delivery_collection, df_spark, 'channel_transcript', 'refund_delivery_collection_response')
    refund_collections_df = df_spark.toPandas()
    return refund_collections_df

def check_refund_promise(df):
    logger.info("Checking refund promise for DataFrame")
    if df.empty:
        df['response_refund_promissed'] = []
        return df
    # results = []
    # for i, row in df.iterrows():
    #     full_prompt = refund_promise_prompt + "\n\nTranscript:\n" + row['channel_transcript']
    #     llm_response = call_llm(full_prompt)
    #     # logger.info(f"Refund promise LLM response for callid {row['callid']}: {llm_response}")
    #     # print(llm_response)
    #     results.append(llm_response)

    # df['response_refund_promissed'] = results

    df_spark = spark.createDataFrame(df)    
    df_spark = call_llm_spark(model, refund_promise_prompt, df_spark, 'channel_transcript', 'response_refund_promissed')
    df = df_spark.toPandas()
    return df

def check_justified_refund_reason(row):
    logger.info("Checking justified refund reason")
    transcript = row['channel_transcript']
    llm_response = call_llm(prompt_justified_refund_reason + "\n\nTranscript:\n" + transcript)
    if "unjustified" in llm_response:
        return False
    else:
        return True

def refund_collections_scoring(refund_collections):
    logger.info("Scoring refund collections")
    refund_delivery_collection_score = []
    refund_delivery_collection_remarks= []
    refund_delivery_collection_comments = []
    if refund_collections.empty:
        refund_collections['refund_delivery_collection_score'] = []
        refund_collections['refund_delivery_collection_remarks'] = []
        refund_collections['refund_delivery_collection_comments'] = []
        return refund_collections
    for i, row in refund_collections.iterrows():
        amnt1, amnt2 = row['adj_amount'], row['value_giftcard']
        amnt_present=False
        amnt_present_below_10 = False
        amnt_present_below_10 = any(float(x) > 0 and float(x) <= 10 for x in str(amnt1).split('\n') + str(amnt2).split('\n') if x.strip().lower() not in ('none', 'nan', ''))
        amnt_present = any(float(x) > 0 for x in str(amnt1).split('\n') + str(amnt2).split('\n') if x.strip().lower() not in ('none', 'nan', ''))
        # logger.info(f"Scoring callid: {row['callid']}")
        # response = check_refund_delivery_collection(row['channel_transcript'])
        response = row['refund_delivery_collection_response']
        # logger.info(f"Refund delivery/collection response: {response}")
        # print(response)
        response = repair_json(response)
        response_json = json.loads(response)
        if response_json['Refund for delivery/collection charges requested'] == 'True':
            refund_reason = response_json['Refund request reason']
            if not 'other' in refund_reason.lower() and amnt_present_below_10:
                refund_delivery_collection_score.append(10)
                refund_delivery_collection_remarks.append('Compliant - Good Outcome')
                refund_delivery_collection_comments.append(f'refunded for {refund_reason}, confidence-High')
            elif 'other' in refund_reason.lower() and amnt_present_below_10:
                justified = check_justified_refund_reason(row)
                if justified==False:
                    refund_delivery_collection_score.append(1)
                    refund_delivery_collection_remarks.append('Non-Compliant No Poor Outcome')
                    refund_delivery_collection_comments.append('delivery or collection charges refunded for unjustified reason, confidence-High')
                else:
                    refund_delivery_collection_score.append(10)
                    refund_delivery_collection_remarks.append('Compliant - Good Outcome')
                    refund_delivery_collection_comments.append(f'confidence-High')
            elif 'true' in row['response_refund_promissed'] and amnt_present==False:
                refund_delivery_collection_score.append(0)
                refund_delivery_collection_remarks.append('Non-Compliant Poor Outcome')
                refund_delivery_collection_comments.append('delivery or collection charges promised but not refunded, confidence-Low')
            else:
                refund_delivery_collection_score.append(10)
                refund_delivery_collection_remarks.append('Compliant - Good Outcome')
                refund_delivery_collection_comments.append('confidence-High')
        else:
            refund_delivery_collection_score.append(10)
            refund_delivery_collection_remarks.append('Compliant - Good Outcome')
            refund_delivery_collection_comments.append('confidence-High')
    refund_collections["refund_delivery_collection_score"] = refund_delivery_collection_score
    refund_collections["refund_delivery_collection_remarks"] = refund_delivery_collection_remarks
    refund_collections["refund_delivery_collection_comments"] = refund_delivery_collection_comments
    logger.info("Refund collections scoring complete")
    return refund_collections

def check_string(text):
    return True if "true" in text.lower() else False

def scoring_nip_inr_wgs(df_nip_inr_wgs):    
    logger.info("Scoring NIP/INR/WGS cases")
    count_refunded = 0
    count_not_refunded = 0
    nip_inr_wgs_score = []
    nip_inr_wgs_remarks = []
    nip_inr_wgs_comments = []
    
    for i,row in df_nip_inr_wgs.iterrows():
        refund_initiated_response = row["response_refund_promissed"]
        refund_initated = check_string(refund_initiated_response)
        if refund_initated:
            amnt1,amnt2 = row['adj_amount'],row['value_giftcard']
            if (amnt1 is not None and amnt1 != '' and not pd.isna(amnt1)) or (amnt2 is not None and amnt2 != '' and not pd.isna(amnt2)):
                # logger.info(f"Refunded for callid: {row['callid']}")
                # print("refunded")
                count_refunded += 1
                nip_inr_wgs_score.append(10)
                nip_inr_wgs_remarks.append('Compliant - Good Outcome')
                nip_inr_wgs_comments.append('refund promised and refunded, confidence-High')
            else:
                # logger.info(f"Not refunded for callid: {row['callid']}")
                # print("not refunded")
                count_not_refunded += 1
                nip_inr_wgs_score.append(0)
                nip_inr_wgs_remarks.append('Non-Compliant Poor Outcome')
                nip_inr_wgs_comments.append('refund promised but not refunded, confidence-High')
        else:
            nip_inr_wgs_score.append(10)
            nip_inr_wgs_remarks.append('Compliant - Good Outcome')
            nip_inr_wgs_comments.append('confidence-High')
    df_nip_inr_wgs["refund_promised_score"] = nip_inr_wgs_score
    df_nip_inr_wgs["refund_promised_remarks"] = nip_inr_wgs_remarks
    df_nip_inr_wgs["refund_promised_comments"] = nip_inr_wgs_comments
    logger.info("NIP/INR/WGS scoring complete")
    return df_nip_inr_wgs

def gogw_scoring(df_gogw):
    logger.info("Scoring GOGW cases")
    gogw_score = []
    gogw_remarks = []
    gogw_comments = []
    llm_response_gogw = []
    for i, row in df_gogw.iterrows():
        values = [v.strip() for v in str(row['value_giftcard']).split('\n') if v.strip() != ""]
        try:
            if any(float(v) > 10 for v in values):
                gogw_llm_prompt = prompt_gogw_check + "\n\nTranscript:\n" + row["channel_transcript"]
                response = call_llm(gogw_llm_prompt)
                llm_response_gogw.append(response)
                # logger.info(f"GOGW LLM response: {response}")
                response = repair_json(response)
                response_json = json.loads(response)
                if response_json['GOGW Promised'] == "True" and response_json['Manager Approval']=="True":
                    gogw_score.append(10)
                    gogw_remarks.append('Compliant - Good Outcome')
                    gogw_comments.append('refund promised and managers approval taken, confidence-High')
                # elif response_json['GOGW Promised'] == "True" and response_json['Manager Approval']=="False":
                #     gogw_score.append(1)
                #     gogw_remarks.append('Non-Compliant No Poor Outcome')
                #     gogw_comments.append('Manager approval not taken, confidence-High')
                else:
                    gogw_score.append(10)
                    gogw_remarks.append('Compliant - Good Outcome')
                    gogw_comments.append('Gogw not promissed, but credited. confidence-Low')
            else:
                gogw_score.append(10)
                gogw_remarks.append('Compliant - Good Outcome')
                gogw_comments.append('confidence-High')
                llm_response_gogw.append("NA")
        except ValueError as e:
            logger.error(f"ValueError in GOGW scoring for callid: {row['callid']}, error: {e}")
            print(f"ValueError in GOGW scoring for callid: {row['callid']}, error: {e}")
            gogw_score.append(10)
            gogw_remarks.append('Compliant - Good Outcome')
            gogw_comments.append('confidence-Low')
            llm_response_gogw.append("NA")
    df_gogw["gogw_score"] = gogw_score
    df_gogw["gogw_remarks"] = gogw_remarks
    df_gogw["gogw_comments"] = gogw_comments
    df_gogw["gogw_llm_response"] = llm_response_gogw
    logger.info("GOGW scoring complete")
    return df_gogw

def amount_reconfirmation_check(merged_df_out_all):
    merged_df_out_all = merged_df_out_all.reset_index(drop=True)
    refund_made = merged_df_out_all[
        (merged_df_out_all['adj_amount'].notnull()) &
        (merged_df_out_all['adj_amount'] != '') &
        (merged_df_out_all['response_refund_promissed'].str.lower().str.contains('true')) &
        (merged_df_out_all['final_score'] == 10)
    ]
    remaining_rows = merged_df_out_all.drop(refund_made.index)
    logger.info(f"Calls with refund made: {refund_made.shape}")
    df_spark_refund_made = spark.createDataFrame(refund_made)
    df_spark_refund_made = call_llm_spark(model, prompt_refund_amount_reconfirmation, df_spark_refund_made, 'channel_transcript', 'refund_confirmation_response')
    refund_made = df_spark_refund_made.toPandas()
    
    refund_made.loc[
        refund_made['refund_confirmation_response'].str.lower().str.contains('false', na=False),
        ['final_score', 'final_remarks', 'final_comments']
    ] = [0, 'Non-Compliant Poor Outcome', 'Agent did not reconfirm the refund amount']
    
    logger.info(f"refund_made Score value counts:\n{refund_made['final_score'].value_counts()}")
    
    merged_df_out_all = pd.concat([remaining_rows, refund_made], ignore_index=True)
    return merged_df_out_all



def get_final_fields(merged_df_out):
    logger.info("Calculating final fields for merged_df_out")
    score_cols = ['price_mismatch_score', 'refund_delivery_collection_score', 'refund_promised_score', 'gogw_score']
    remarks_cols = ['price_mismatch_remarks', 'refund_delivery_collection_remarks', 'refund_promised_remarks', 'gogw_remarks']
    comments_cols = ['price_mismatch_comments', 'refund_delivery_collection_comments', 'refund_promised_comments', 'gogw_comments']
    for c in score_cols + remarks_cols + comments_cols:
        if c not in merged_df_out.columns:
            merged_df_out[c] = np.nan
    merged_df_out['score'] = merged_df_out[score_cols].values.tolist()
    merged_df_out['remarks'] = merged_df_out[remarks_cols].values.tolist()
    merged_df_out['comments'] = merged_df_out[comments_cols].values.tolist()

    final_score = []
    final_remarks = []
    final_comments = []
    for _, row in merged_df_out.iterrows():
        scores = [s for s in row['score'] if s not in [None, '', np.nan] and not (isinstance(s, float) and np.isnan(s))]
        if not scores:
            return (None, None, None)
        min_score = min(scores)
        idx = row['score'].index(min_score)
        final_remark = row['remarks'][idx] if idx < len(row['remarks']) else None
        final_comment = row['comments'][idx] if idx < len(row['comments']) else None
        final_score.append(min_score)
        final_remarks.append(final_remark)
        final_comments.append(final_comment)
    
    merged_df_out[['final_score', 'final_remarks', 'final_comments']] = pd.DataFrame({'final_score': final_score, 'final_remarks': final_remarks, 'final_comments': final_comments})
    logger.info("Final fields calculation complete")
    return merged_df_out

def confidence_score(df):
    confidences = []
    new_rows = []
    for i,row in df.iterrows():
        comments = row['final_comments']
        if comments:
            if 'confidence' in str(comments):
                conf = comments.split("confidence")
                # print(conf)
                confidences.append(conf)
                row['final_comments'] = conf[0].strip()
                row['confidence'] = conf[1].strip().strip('-')
                new_rows.append(row)
            else:
                row['confidence'] = ''
                new_rows.append(row)
        else:
            row['confidence'] = ''
            new_rows.append(row)
    df_out = pd.DataFrame(new_rows)
    return df_out

def save_to_db(catalog_config, formatted_date, final_results):
    filtered_table = catalog_config['intermediate']['refund_gogw']
    try:
        date_exists = (
            spark.table(filtered_table)
            .filter(col("call_date") == formatted_date)
            .limit(1)
            .count() > 0
        )
        if date_exists:
            logger.info(f"Results for {formatted_date} already exists. Deleting old data before insert.")
            spark.sql(f"DELETE FROM {filtered_table} WHERE call_date = '{formatted_date}'")
    except Exception as e:
        logger.error(f"Error in deleting old data: {e}, continuing with the insert") 
    final_results.write.format("delta").option("mergeSchema","true").mode("append").saveAsTable(catalog_config['intermediate']['refund_gogw'])

if __name__ == "__main__":
    # call_date = '2026-03-14'
    try:
        # call_date = dbutils.jobs.taskValues.get(taskKey="Pre-Modules-Run", key="sql_date")
        call_date = '2026-06-21'
    except Exception as e:
        call_date = sys.argv[1]
        if not re.match(r"^\d{4}-\d{2}-\d{2}$", str(call_date)):
            raise ValueError(f"formatted_date '{call_date}' is not in yyyy-mm-dd format")
        logger.info(f"formatted_date: {call_date}")
    catalog_config = yaml.safe_load(open('../catalog_config.yaml', 'r'))
    main_config = yaml.safe_load(open('../main_config.yaml', 'r'))

    logger.info(f"Starting main process for date: {call_date}")
    # adjustment_path= "/Workspace/root/ContactCentre/iAudit Processes/Data/GOGW and Adjustment for iAudit - Dec-25 Onwards v2.xlsx"
    # adjustment_path= "/Workspace/Users/pawan_kumar@next.co.uk/pawan_dev/refund_applied_correctly/data/merged_jan_feb_and_gift.xlsx"
    # adjustment_sheet= "Adjustments_jan_feb"
    # gift_card_sheet= "GiftCards_jan_feb"
    # adjustment_path= "/Workspace/Users/pawan_kumar@next.co.uk/pawan_dev/refund_applied_correctly/data/GOGW and Adjustment for iAudit - Feb-Mar-26 (R3).xlsx"
    # adjustment_sheet= "Adjusments logged Mar-26"
    # gift_card_sheet= "Gift Cards logged Mar-26"
    adjustment_path = main_config['File_Paths']['adjustment_path']
    adjustment_sheet = main_config['adjustment_sheet']
    gift_card_sheet = main_config['gift_card_sheet']

    df = spark.sql(f"""
        SELECT *
        -- FROM contactcentre_prod.transcripts.transcripts_raw
        FROM contactcentre_prod.iaudit.filtered_input_data_v2_iaudit
        WHERE DATE(call_date) = '{call_date}'
    """).toPandas()

    # call_ids = [
    #     "fb02e062-7250-40d7-8f7f-9b8cbbf7a9a0",
    #     "045d5511-8c6a-4239-87b6-44a5036d940d",
    #     "0a4a3647-47ec-441a-8bef-e9a911a451d5",
    #     "e143af75-381f-46d8-879c-bc3e708a4890",
    #     "37398665-50f2-423f-b56f-bfe488616ef1",
    #     "2e1a1665-c504-4b16-8c14-c8519a388ed2",
    #     "91e35072-32dc-40eb-b13f-0d083fb019df"
    # ]

    # call_ids_str = ",".join([f"'{cid}'" for cid in call_ids])
    # df = spark.sql(f"""
    #     SELECT *
    #     FROM contactcentre_prod.transcripts.transcripts_raw
    #     WHERE callid IN ({call_ids_str})
    # """).toPandas()

    # print("input transcript rows: ",df.shape)
    logger.info(f"Input transcript rows: {df.shape}")
    if 'AgentId' in df.columns:
        df = df.rename(columns={'AgentId': 'AgentID'})
    merged_df = preprocess_transcript(df)
    call_to_run = int(len(merged_df) * 0.1)
    merged_df = merged_df.head(call_to_run)
    merged_df = merged_df.copy()
    print(merged_df.shape)
    logger.info(f"Merged transcript shape: {merged_df.shape}")
    df_prev_caller, merged_df = split_last_prev_callers(merged_df)
    print(df_prev_caller.shape, merged_df.shape)
    logger.info(f"Prev caller shape: {df_prev_caller.shape}, merged_df shape: {merged_df.shape}")

    # top_10_percent = int(len(merged_df) * 1)
    # merged_df = merged_df.head(top_10_percent)
    # subset_df = merged_df

    logger.info("Step 2: Fetching logger data from zen_live9")
    logger_table_pandas = get_logger_data(merged_df)
    print("logger_data.shape before aggregation",logger_table_pandas.shape)
    logger.info(f"Logger data shape before aggregation: {logger_table_pandas.shape}")
    logger_table_pandas = logger_table_pandas.groupby(['conversation_id', 'account_number'], as_index=False).agg({
        'ResponseName': lambda x: ',\n'.join(str(i) if i is not None else '' for i in x),
        'call_date': 'first'
    })
    print("logger_data.shape after aggregation",logger_table_pandas.shape)
    logger.info(f"Logger data aggregated, final shape: {logger_table_pandas.shape}")

    merged_logger = merged_df.merge(logger_table_pandas, left_on='callid', right_on='conversation_id', how='left').drop(columns=['conversation_id'])
    merged_giftcard = read_adjustment_data(adjustment_path, adjustment_sheet, gift_card_sheet)

    #payin3 account details
    merged_giftcard = fetch_payin3_accounts(merged_giftcard)
    # df_account_details = fetch_payin3_accounts(merged_giftcard)

    # df_payin3 = merged_giftcard[merged_giftcard['AccountType'].str.contains('PayIn3', na=False)]
    df_payin3 = merged_giftcard[
        merged_giftcard['AccountType'].str.contains('PayIn3', na=False) | (merged_giftcard['payin3_queue'] == True)
    ]
    merged_giftcard = merged_giftcard[~(merged_giftcard['AccountType'].str.contains('PayIn3', na=False) | (merged_giftcard['payin3_queue'] == True))]
    logger.info(f"Shape of payin3 calls: {df_payin3.shape}")
    logger.info(f"Shape of non payin3 calls: {merged_giftcard.shape}")

    # Price Match calls processing and scoring
    price_match_calls = merged_giftcard[merged_giftcard['ResponseName'].str.contains("Pricing/Multi Buy Query", na=False)]
    logger.info(f"Price match calls shape: {price_match_calls.shape}")
    price_match_calls = process_price_mismatch(price_match_calls)

    # Refund delivery collection calls processing and scoring
    refund_collections = merged_giftcard[merged_giftcard['ResponseName'].str.contains('Delivery Charge|Collection charge', na=False)]
    logger.info(f"Refund collections shape: {refund_collections.shape}")
    refund_collections = check_refund_delivery_collection_response(refund_collections)
    refund_collections = check_refund_promise(refund_collections)
    refund_collections = refund_collections_scoring(refund_collections)

    # refund_inr_nip processing and scoring
    df_nip_inr_wgs = merged_giftcard[
        (
            merged_giftcard['ResponseName'].str.contains('Not in parcel billed', na=False) |
            merged_giftcard['ResponseName'].str.contains('Not in parcel not billed', na=False) |
            merged_giftcard['ResponseName'].str.contains('INR', na=False) |
            merged_giftcard['ResponseName'].str.contains('Wrong Goods Sent', na=False)
        )
    ]
    logger.info(f"NIP/INR/WGS shape: {df_nip_inr_wgs.shape}")

    df_nip_inr_wgs = check_refund_promise(df_nip_inr_wgs)
    df_nip_inr_wgs = scoring_nip_inr_wgs(df_nip_inr_wgs)

    ## GOGW processing and scoring
    df_gogw = merged_giftcard[(merged_giftcard['value_giftcard'].notnull()) & (merged_giftcard['value_giftcard'] != "")]
    logger.info(f"GOGW shape: {df_gogw.shape}")
    df_gogw = gogw_scoring(df_gogw)

    # Merge DataFrames on 'pk', adding only new columns from each subsequent DataFrame
    merged_df_out = price_match_calls.copy()

    for df in [refund_collections, df_nip_inr_wgs, df_gogw]:
        merged_df_out = (
            merged_df_out.set_index("pk")
            .combine_first(df.set_index("pk"))
            .reset_index()
        )
        logger.info(f"merged_df_out shape: {merged_df_out.shape}")

    # print(merged_df_out.columns)
    merged_df_out = get_final_fields(merged_df_out)
    
    # merged_df_out.to_csv("data/merged_df_out_audited.csv",index=False)
    df_prev_caller['final_score'] = 10
    df_prev_caller['final_remarks'] = "Compliant - Good Outcome"
    df_prev_caller['final_comments'] = "1st agent, transferred call"

    df_payin3["final_score"] = 10
    df_payin3["final_remarks"] = "Compliant Good Outcome"
    df_payin3["final_comments"] = "Manual validation is required to confirm the refund/adjustment status for Pay in 3 accounts, as  information is not available in the database."

    # print(merged_df_out['pk'])
    merge_df_remaining = merged_giftcard[~merged_giftcard['pk'].isin(merged_df_out['pk'])]
    merge_df_remaining['final_score'] = 10
    merge_df_remaining['final_remarks'] = "Compliant - Good Outcome"
    merge_df_remaining['final_comments'] = "NA"

    merged_df_out_all = pd.concat([merged_df_out,df_prev_caller, df_payin3, merge_df_remaining])

    merged_df_out_all = amount_reconfirmation_check(merged_df_out_all)
    merged_df_out_all.to_excel(f"/Workspace/Users/pawan_kumar@next.co.uk/refund_gogw_prod_setup/merged_df_intermediate_{call_date}.xlsx", index=False)

    print(merged_df_out_all.shape)
    cols_to_drop = [
        "call_date", "account_type_adj", "adj_type", "adj_reason", "adj_reason_code",
        "subcategory_adj", "item_number_adj", "item_description_adj", "item_size_adj",
        "return_method_adj", "reason_giftcard", "currency_giftcard", "year", "week", "type_giftcard", 
        "mismatch_issue", "type_of_mismatch", "evidence", "authorising_tm", "brand",
        "channel_transcript", "score", "remarks", "comments"
    ]
    merged_df_out_all = merged_df_out_all.drop(columns=cols_to_drop)

    logger.info(f"Final merged output shape before sorting: {merged_df_out_all.shape}")
    merged_df_out_all = merged_df_out_all.reset_index(drop=True)
    merged_df_out_all = merged_df_out_all.sort_values(['pk', 'final_score'], ascending=[True, True])
    merged_df_out_all = merged_df_out_all.drop_duplicates(subset='pk', keep='first')
    print(merged_df_out_all.shape)
    logger.info(f"Final merged output shape after deduplication: {merged_df_out_all.shape}")

    merged_df_out_all = confidence_score(merged_df_out_all)
    merged_df_out_all = merged_df_out_all.rename(columns={'final_score': 'Score', 'final_remarks': 'Result', 'final_comments': 'Comment'})
    logger.info(f"Score value counts:\n{merged_df_out_all['Score'].value_counts()}")
    merged_df_out_all.to_excel(f"/Workspace/Users/pawan_kumar@next.co.uk/refund_gogw_prod_setup/merged_df_out_{call_date}.xlsx", index=False)

    merged_df_out_all['call_date'] = pd.to_datetime(call_date)
    final_result_spark = spark.createDataFrame(merged_df_out_all)

    # Cast columns

    column_dtype_map = {
        "pk": StringType(),
        "AccountType": StringType(),
        "AgentID": StringType(),
        "ResponseName": StringType(),
        "SpecialAccountDescription": StringType(),
        "account_number": StringType(),
        "adj_amount": DoubleType(),
        "callid": StringType(),
        "createdate": TimestampType(),
        "gogw_comments": StringType(),
        "gogw_llm_response": StringType(),
        "gogw_remarks": StringType(),
        "gogw_score": DoubleType(),
        "price_mismatch_comments": StringType(),
        "price_mismatch_remarks": StringType(),
        "price_mismatch_response": StringType(),
        "price_mismatch_score": DoubleType(),
        "refund_delivery_collection_comments": StringType(),
        "refund_delivery_collection_remarks": StringType(),
        "refund_delivery_collection_response": StringType(),
        "refund_delivery_collection_score": DoubleType(),
        "refund_promised_comments": StringType(),
        "refund_promised_remarks": StringType(),
        "refund_promised_score": DoubleType(),
        "response_refund_promissed": StringType(),
        "value_giftcard": StringType(),
        "Score": DoubleType(),
        "Result": StringType(),
        "Comment": StringType(),
        "confidence": StringType(),
        "call_date": TimestampType(),
    }

    # Only cast columns that exist in the DataFrame
    for col_name, col_type in column_dtype_map.items():
        if col_name in final_result_spark.columns:
            final_result_spark = final_result_spark.withColumn(col_name, F.col(col_name).cast(col_type))


    # save_to_db(catalog_config, call_date, final_result_spark)
    logger.info("Process complete. Output written to data/merged_df_out_all.csv")