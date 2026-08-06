import subprocess
subprocess.check_call(['pip', 'install', 'json-repair', '-q'])
import os,sys
sys.path.append('/Workspace/root/ContactCentre/iAudit Processes/iaudit_chat')

import warnings
import os
import json
from json_repair import repair_json
import re
from datetime import datetime
import numpy as np
import pandas as pd
from pyspark.errors import PySparkException
from pyspark.sql.types import StructType, StructField, BooleanType, StringType
from pyspark.sql.functions import from_json, col, lit, udf, when, expr
from utils.iaudit_logger import get_logger
from chat_prompt_InformationStatementExplanationPrompts import *
from pyspark.sql.functions import format_string
from pyspark.sql.functions import array, lit, when, size, concat_ws
from pyspark.sql import functions as F
import yaml


warnings.filterwarnings("ignore")

# transcript_path = r'/Workspace/Users/ananthagiri_abhiram@next.co.uk/iAudit_Trisha/NonRedactedUAT.xlsx'
# /Workspace/Users/ananthagiri_abhiram@next.co.uk/iAudit_Trisha/infoMarkdown0612.xlsx
# transcript_path = r'/Workspace/Users/ananthagiri_abhiram@next.co.uk/iAudit_Trisha/AuditedInformationCreditTranscipts.xlsx'
#pre audited calls``
# transcript_path = r'/Workspace/Users/ananthagiri_abhiram@next.co.uk/iAudit_Trisha/AllAuditedTransciptNonRedacted2026.xlsx'

# #one day call
# transcript_path = r'/Workspace/Users/ananthagiri_abhiram@next.co.uk/iAudit_Trisha/combined_file_2026_03_27.xlsx'

result_path = r"xInformationExplanationExplanationFinalResults.xlsx"
log = get_logger()

def safe_repair_and_load(s):
    try:
        repaired = repair_json(s)
        return json.loads(s)
    except Exception:
        return None
    
priority = ["Red Error", "Amber Error", "Compliant with development"]
def get_result_priority(row):
    vals = [v for v in row if v!= "N.A."]
    if len(vals) == 0:
        return "N.A."
    for p in priority:
        if p in vals:

            return p
    return "Good Customer Outcome"  

score_map = {

"Red Error": 0,
"Amber Error": 1,
"Compliant with development": 1,
"Good Customer Outcome": 10,
 "N.A.": 10

}
def select_row_effect_end_date_condition(group):
    #case1 if any blank
     if group['EffectiveEndDate'].isna().any():
         return group[group['EffectiveEndDate'].isna()]
     else:
         #case 2 all have values then return latest
         return group.loc[group['EffectiveEndDate'].idxmax()].to_frame().T



if __name__=="__main__":
    try:
        # Read from Unity Catalog table as input, filtered by call_date
        formatted_date = '2026-07-29'
        INPUT_FILE_PATH = spark.table('contactcentre_prod.iaudit.filtered_input_data_iaudit_chat_v2').filter(
            F.col('call_date') == formatted_date
        )
        transcripts = INPUT_FILE_PATH.toPandas()

        # Rename columns to match downstream pipeline expectations
        transcripts = transcripts.rename(columns={
            'conversation_id': 'callid',
            'agent_id': 'AgentId',
            'convo': 'transcript',
            'call_date': 'createdate'
        })
        print(transcripts.columns)
        print(transcripts.shape)
        print(transcripts.head(2))
        # Add channel column for chat data
        # transcripts['channel'] = 'chat'


        total_calls = len(transcripts['callid'].unique().tolist())
        filtered_calls = transcripts['callid'].unique().tolist()[:total_calls]
        # filtered_calls = transcripts['callid'].unique().tolist()[:100]
        df_transcripts = transcripts[transcripts['callid'].isin(filtered_calls)].reset_index(drop=True)
    
        audit_callids = df_transcripts['callid'].unique().tolist()
        print(df_transcripts['callid'].head())

        # print(df_logger['CallId'].head())

        print(df_transcripts['callid'].dtype)

        
        # df_transcripts['createdate'] = pd.to_datetime(df_transcripts['createdate'], format = 'mixed', errors='coerce')
        # df_transcripts = df_transcripts.sort_values(by=["callid", 'createdate'])
        # df_transcripts['createdate'] = df_transcripts['createdate'].astype(str)

        spark_logger =  spark.sql(f"""
                SELECT * FROM contactcentre_prod.staging.zen_live9 
                WHERE conversation_id IN {tuple(audit_callids)}      
            """)

        df_logger = spark_logger.toPandas()

        d1 = {'id': 'ticket_id',
            'Classification': '',
            'call_date': '',
            'channel': 'Channel_logger',
            'CallResolution': '',
            'CallType': '',
            'DepartmentName': '',
            'WorkGroupName': 'Queues',
            'IsAuditable': '',
            'TreeVersion': '',
            'isexpressionofdissatisfaction': '',
            'account_number': '',
            'account_type': '',
            'ICW_reference': '',
            'assignee': '',
            'conversation_id': 'CallId',
            'CountryName': '',
            'Supplier': '',
            'Adjustment_Value': '',
            'Compensation_value': '',
            'Vulnerable_customer_caseid': 'Vulnerable Customer Case ID',
            'Vulnerable_Customer': 'Vulnerable Customer',
            'RootResponseName': 'Level 1',
            'ResponseName': 'Level 2',
            'ReasonL1': '',
            'ReasonL2': '',
            'RootResponseName1': '',
            'ResponseName1': '',
            'Vulnerable_Customer1_a': 'Vuln Support Offered',
            'tTalk': '',
            'tHold': '',
            'tAcw': '',
            'created_at': '',
            'Sub_Department': ''}

        rename_cols = {k:v for k,v in d1.items() if v!=''}
        df_logger = df_logger.rename(columns=rename_cols)
        # Normalize Level 2 values
        df_logger['Level 2'] = (
            df_logger['Level 2']
            .astype(str)
            .str.strip()
            .str.replace('\u00a0', ' ', regex=False)
        )
        print("Logger unique callids", len(df_logger['CallId'].unique().tolist()))
        customer_account_number_list = df_logger['account_number'].unique().tolist()
        customer_account_number_list = [x for x in customer_account_number_list if str(x) != 'nan']
        print("unique_account_number",len(customer_account_number_list))
        customer_account_number_list = [ x for x in customer_account_number_list if x is not None]
        print("Not none unique_account_number",len(customer_account_number_list))
        print("only logger shape",df_logger.shape )

        spark_vv_statement_detail =  spark.sql(f"""
                    SELECT `AccountNumber`, `StatementDate`, `LastPaymentDate`, `LastPaymentValue`, `LastStatementNumber`
    
        FROM 
        `contactcentre_prod`.`iaudit`.`vw_statement_detail`
        WHERE 
        (`AccountNumber`, `LastPaymentDate`) IN (
            SELECT 
            `AccountNumber`, 
            MAX(`LastPaymentDate`) AS LatestPaymentDate
            FROM 
            `contactcentre_prod`.`iaudit`.`vw_statement_detail`
            WHERE 
            `AccountNumber` IN {tuple(customer_account_number_list)}
            GROUP BY 
            `AccountNumber`
        )     
                    """)

        df_statement_detail = spark_vv_statement_detail.toPandas()
        print("statement_data",df_statement_detail.shape)
        df_statement_detail = df_statement_detail.rename(columns={'AccountNumber': 'account_number'})
        df_logger = df_logger.merge(df_statement_detail, on='account_number', how='left')
        df_logger['account_number'] = df_logger['account_number'].astype(str)
        # print("with statement detail",df_logger.shape)
        

###########################################################################################        

        spark_vw_customers = spark.sql(f"""
        SELECT `AccountNumber`, `EffectiveEndDate`, 
            `StatementDate`,
            `LastStatementNumber`,
            `NextStatementDate`, 
            `CreditLimit`
        FROM `businessintelligencesystems_prod`.`online_pii_transformed`.`vw_customers`
        WHERE `AccountNumber` IN {tuple(customer_account_number_list)}  
    """)
        df_vw_customers= spark_vw_customers.toPandas()
        df_vw_customers = df_vw_customers.rename(columns={'AccountNumber': 'account_number'})
        print("df_vw_customers", df_vw_customers.shape)
        df_vw_customers['EffectiveEndDate']= pd.to_datetime( df_vw_customers['EffectiveEndDate'], errors ='coerce')
        df_vw_customers= df_vw_customers.groupby('account_number', group_keys=False).apply(select_row_effect_end_date_condition).reset_index( drop=True)
        # print('df_vw_customers after',df_vw_customers.shape)
        df_vw_customers['NextStatementNumber'] =  df_vw_customers['LastStatementNumber'] + 1
       

        df_logger = df_logger.merge(df_vw_customers, on='account_number', how='left')
        df_logger['account_number'] = df_logger['account_number'].astype(str)



###################################################################################
        # print("with customer data",df_logger.columns)
        # print("with customer data",df_logger.shape)
        # print("customer data",df_vw_customers.shape)
        # print("customer data unique account_number",len(df_vw_customers['account_number'].unique().tolist()))

        # Last Order Date
        spark_vw_orders = spark.sql(f"""
            SELECT
            `CustomerNo`,
            `OrderDate` as `LastOrderDate`, `DeliveryCharge`
        FROM
            `contactcentre_prod`.`iaudit`.`vw_order_details`
        WHERE
            `CustomerNo` IN {tuple(customer_account_number_list)}
            AND `OrderDate` IN (
                SELECT MAX(`OrderDate`)
                FROM `contactcentre_prod`.`iaudit`.`vw_order_details` AS sub
                WHERE sub.`CustomerNo` = `vw_order_details`.`CustomerNo`
            )  
        """)
        df_vw_orders= spark_vw_orders.toPandas()
        df_vw_orders = df_vw_orders.rename(columns={'CustomerNo': 'account_number'})
        d_vw_orders = df_vw_orders.drop_duplicates(subset=['account_number'], keep='first')
        #df_vw_orders.to_excel('orderscheck.xlsx')


        df_logger = df_logger.merge(df_vw_orders, on='account_number', how='left')
        df_logger['account_number'] = df_logger['account_number'].astype(str)
        # print("with order data",df_logger.columns)
        # print("with order data",df_logger.shape)



        level_2_req_list = ["Account Balance Request",
            "Collection Charge",
            "Delivery Charge",
            "Incorrect Statement Received",
            "Interest Charge",
            "Return Charge",
            "Statement Not Received/Viewable", "Place an order ",
            "Arrange a collection",
            "Arrange A Collection",
            "Order cancellation ",
            "Make Payment",
            "Place An Order"

    ] 
        level_2_req_list = [
            x.strip().replace('\u00a0', ' ')
            for x in level_2_req_list
        ]

        print(df_logger['Level 2'].value_counts(dropna=False).head(50))

        
        df_logger['priority'] =df_logger['Level 2'].isin(level_2_req_list).astype(int)
        print(df_logger['priority'].value_counts(dropna=False))
        
        # print("Logger before",df_logger.shape)
        df_logger_unique = (df_logger.sort_values(['CallId', 'priority'], ascending=[True, False]).drop_duplicates(subset = 'CallId', keep='first').drop(columns='priority'))
        # print(df_logger_unique.shape)
        print(df_logger.columns.tolist())

        for col in df_logger.columns:
            print(col, df_logger[col].notna().sum())
        print(df_logger.head())
        data = pd.merge(df_transcripts, df_logger_unique, left_on='callid', right_on='CallId', how='left')

        # data['callid_agentid']= data['callid'] + "|" + data['AgentId']
        print("data",data.columns)
        # 
        # grouped_transcript_all = data.groupby('callid').apply(
        #     lambda x: "\n".join(
        #         f"{row['transcript']}"
        #         for _, row in x.iterrows()
        #     )
        # ).reset_index(name='transcript')
        # data_intermediate = data.groupby('callid', as_index = False).first()
        # data_req = pd.merge(df_transcript, data_intermediate, on='callid', how='left')
        data_req = data.copy()
        data_req['createdate'] = pd.to_datetime(data_req['createdate'], format = 'mixed', errors='coerce')
        data_req = data_req.sort_values(by=["callid", 'createdate'], ascending =[True, True])
        data_req['createdate'] = data_req['createdate'].astype(str)
        data_req['agent_order'] = 1
        data_req['agent_order_max'] = 1

        #########################################################################
        data_req['StatementDate_y'] = pd.to_datetime(data_req['StatementDate_y'], errors='coerce')
        data_req['required_payment_due_date'] = data_req['StatementDate_y'] + pd.DateOffset(days= -7)
        print("final data",data_req.shape)
        print(data_req['Level 2'].value_counts(dropna=False).head(50))
        #data_req.to_excel("check_data_information_statement.xlsx")
        
        # data_
        
        # req = pd.read_excel(r"/Workspace/Users/ananthagiri_abhiram@next.co.uk/iAudit_Trisha/check_data_information_statement.xlsx")

        acccount_balance_list = ["Account Balance Request",
        "Incorrect Statement Received",
        "Interest Charge",
        "Return Charge",
        "Statement Not Received/Viewable",
        ]



        place_order_list = ["Place an order", "Delivery Charge", "Make Payment", "Place An Order"]

        collection_list = ["Arrange a collection", "Arrange A Collection", "Collection Charge"]

        level_2_req_list = list(set(acccount_balance_list + place_order_list + list(collection_list)))

        data_req['account_balance_enq_indicator'] = np.where(data_req['Level 2'].isin(acccount_balance_list), 1,0)
        data_req['place_order_indicator'] = np.where(data_req['Level 2'].isin(place_order_list), 1,0)
        data_req['collection_indicator'] = np.where(data_req['Level 2'].isin(collection_list), 1,0)
        data_req['account_information'] = 1
        df_prompts = (
                            spark.createDataFrame(data_req).withColumn(
                                "user_prompt", F.when((F.col("account_balance_enq_indicator")==1) , F.lit(user_prompt_account_balance))
                                .when((F.col("place_order_indicator")==1), F.lit(user_prompt_place_an_order))
                                .when((F.col("collection_indicator")==1), F.lit(user_prompt_arrange_a_collection))
                                .otherwise(F.lit(""))

                            )
                            .withColumn("user_prompt1", F.when((F.col("account_information")==1), F.lit(user_prompt_account_information))
                            .otherwise(F.lit(""))

                            )

                            .withColumn("final_prompt", 
                                        F.when( (F.col("account_balance_enq_indicator")==1),
                                                F.concat_ws( "\n\n",
                                                            F.lit("SYSTEM PROMPT:"),
                                                            F.lit(system_prompt),
                                                            F.lit("USER PROMPT:"),
                                                            F.col("user_prompt"),
                                                            F.lit("TRANSCRIPT:"),
                                                            F.col("transcript"),
                                                            F.lit("Information from Database:"),
                                                            F.lit("Credit Limit:"),
                                                            F.col("CreditLimit"),
                                                            F.lit("Last Statement Number:"),
                                                            F.col("LastStatementNumber_y"),
                                                            F.lit("Last Statement Date:"),
                                                            F.col("StatementDate_y"),
                                                            F.lit("Next Statement Number:"),
                                                            F.col("NextStatementNumber"),
                                                            F.lit("Next Statement Date:"),
                                                            F.col("NextStatementDate"),
                                                            F.lit("Last Payment Amount:"),
                                                            F.col("LastPaymentValue"),
                                                            F.lit("Last Payment Date:"),
                                                            F.col("LastPaymentDate"),
                                                            F.lit("Last Order Date"),
                                                            F.col("LastOrderDate"),
                                                            F.lit("Required Payment Due Date:"),
                                                            F.col("required_payment_due_date"),
                                                            F.lit("Call date:"),
                                                            F.col("call_date")
                                                

                                                            )
                                                ) .otherwise(
                                                    F.concat_ws( "\n\n",
                                                            F.lit("SYSTEM PROMPT:"),
                                                            F.lit(system_prompt),
                                                            F.lit("USER PROMPT:"),
                                                            F.col("user_prompt"),
                                                            F.lit("TRANSCRIPT:"),
                                                            F.col("transcript")
                                                
                                                            
                                                            )
                                                )
                                                )
                                                .withColumn("final_prompt1", F.when((F.col("account_information")==1),
                                                        F.concat_ws( "\n\n",
                                                            F.lit("SYSTEM PROMPT:"),
                                                            F.lit(system_prompt),
                                                            F.lit("USER PROMPT:"),
                                                            F.col("user_prompt1"),
                                                            F.lit("TRANSCRIPT:"),
                                                            F.col("transcript")
                                                
                                        
                                                            )))
                            

            )
        
        # databricks-meta-llama-3-1-8b-instruct
        df_ai = (
            df_prompts.withColumn(
                "ai_response", F.when(
                    (F.col("Level 2").isin(level_2_req_list)) &
                (F.col("agent_order_max")== F.col("agent_order")) &
                (F.col("user_prompt") != ""), F.expr(
                    """ai_query('contact_centre_internal_batch',request => final_prompt)"""
                        )
            )
            
                )
        
        
            )


        # df_ai = (
        #     df_ai1.withColumn(
        #             "ai_response1", F.when(
        #             # 	(F.col("Level 2").isin(level_2_req_list)) &
        #             # (F.col("agent_order_max")== F.col("agent_order")) &
        #             (F.col("user_prompt1") != ""), F.expr(
        #                 """ai_query('contact_centre_internal_batch',request => final_prompt1)"""
        #                     )
        #         )
                
        #     )
            
            
        #     )
            
        #display(df_ai.toPandas())
            
            
        df_ai_pandas = df_ai.toPandas()
        print("LLM calls done", len(df_ai_pandas))
        df_ai_pandas.to_excel("InformationStatementExplanation29.xlsx")
        if 'ai_response' not in df_ai_pandas.columns:
            df_ai_pandas['ai_response'] = ""
        # if 'ai_response1' not in df_ai_pandas.columns:
        #     df_ai_pandas['ai_response1'] = ""    
        
        df_ai_pandas['ai_response2'] = (df_ai_pandas['ai_response']
                                        .str.replace("```json", "", regex = False)
                                        .str.replace("```", "",regex = False)
                                        .str.strip()
                                        )
        
        # df_ai_pandas['ai_response3'] = (df_ai_pandas['ai_response1']
        #                                 .str.replace("```json", "", regex = False)
        #                                 .str.replace("```", "",regex = False)
        #                                 .str.strip()
        #                                 )								
        df_ai_pandas['Output_final'] = df_ai_pandas['ai_response2'].apply(safe_repair_and_load)
        df_ai_pandas = pd.concat([df_ai_pandas,df_ai_pandas['Output_final'].apply(pd.Series)], axis =1)




        # ======================================================
        # INTERMEDIATE DATABASE + TRANSCRIPT VALIDATION EXPORT
        # ======================================================

        print("\nCreating intermediate validation excel...\n")

        validation_export_cols = [

            # identifiers
            'callid',
            'Level 2',

            # final audit result
            'iAudit Result: Information Statement Explanation',

            # transcript
            'transcript',

            # database fields
            'CreditLimit',
            'LastStatementNumber_y',
            'StatementDate_y',
            'NextStatementNumber',
            'NextStatementDate',
            'LastPaymentValue',
            'LastPaymentDate',
            'LastOrderDate',
            'required_payment_due_date'
        ]

        # keep only columns that actually exist
        validation_export_cols = [
            col for col in validation_export_cols
            if col in df_ai_pandas.columns
        ]

        validation_export_df = df_ai_pandas[
            validation_export_cols
        ].copy()

        # # optional filter only account balance request
        # validation_export_df = validation_export_df[
        #     validation_export_df['Level 2']
        #     ==
        #     'Account Balance Request'
        # ]

        # export
        validation_export_df.to_excel(
            "Intermediate_Validation29.xlsx",
            index=False
        )

        print(
            "Intermediate validation file created successfully"
        )

        print(validation_export_df.head())
                # =========================================================
        # INTERMEDIATE VALIDATION BLOCK
        # LLM vs TRANSCRIPT vs DATABASE
        # =========================================================

        print("\n================ VALIDATION START ================\n")

        # ---------------------------------------------------------
        # COPY DATAFRAME
        # ---------------------------------------------------------

        validation_df = df_ai_pandas.copy()

        print("Initial validation_df shape:", validation_df.shape)

        # ---------------------------------------------------------
        # FILTER ONLY ACCOUNT BALANCE REQUEST
        # ---------------------------------------------------------

        validation_df = validation_df[
            validation_df['Level 2']
            ==
            'Account Balance Request'
        ].copy()

        print("Account Balance rows:", validation_df.shape)

        # ---------------------------------------------------------
        # CHECK AVAILABLE COLUMNS
        # ---------------------------------------------------------

        print("\nAvailable columns:\n")
        print(validation_df.columns.tolist())

        # ---------------------------------------------------------
        # SAFE EXTRACTION FUNCTION
        # ---------------------------------------------------------

        def extract_credit_limit(text):

            try:

                if pd.isna(text):
                    return None

                text = str(text).lower()

                patterns = [

                    r'credit limit (?:is|of)?\s*[£]?\s*(\d+[,\d]*)',

                    r'limit (?:is|of)?\s*[£]?\s*(\d+[,\d]*)',

                    r'credit limit available (?:is)?\s*[£]?\s*(\d+[,\d]*)'
                ]

                for pattern in patterns:

                    match = re.search(pattern, text)

                    if match:

                        value = (
                            match.group(1)
                            .replace(',', '')
                            .strip()
                        )

                        return float(value)

                return None

            except Exception as e:

                print("Extraction error:", e)

                return None


        # ---------------------------------------------------------
        # EXTRACT CREDIT LIMIT FROM TRANSCRIPT
        # ---------------------------------------------------------

        validation_df['transcript_credit_limit'] = (
            validation_df['transcript']
            .apply(extract_credit_limit)
        )

        print("\nTranscript extraction completed")

        # ---------------------------------------------------------
        # DATABASE CREDIT LIMIT
        # ---------------------------------------------------------

        validation_df['db_credit_limit'] = pd.to_numeric(
            validation_df['CreditLimit'],
            errors='coerce'
        )

        print("\nDatabase conversion completed")

        # ---------------------------------------------------------
        # ACTUAL MATCH LOGIC
        # ---------------------------------------------------------

        validation_df['actual_match'] = np.where(

            (
                validation_df['transcript_credit_limit'].notna()
            )
            &
            (
                validation_df['db_credit_limit'].notna()
            )
            &
            (
                abs(
                    validation_df['transcript_credit_limit']
                    -
                    validation_df['db_credit_limit']
                ) <= 5
            ),

            "Yes",

            "No"
        )

        print("\nActual match calculation completed")

        # ---------------------------------------------------------
        # LLM COLUMN NAME
        # ---------------------------------------------------------

        llm_col = (
            "Was the credit limit provided by the agent correct as per database?"
        )

        print("\nChecking LLM column existence...")

        # ---------------------------------------------------------
        # SAFE LLM COLUMN HANDLING
        # ---------------------------------------------------------

        if llm_col not in validation_df.columns:

            print(f"\nWARNING: {llm_col} column NOT FOUND")

            validation_df[llm_col] = np.nan

        else:

            print(f"\nLLM column FOUND")

        # ---------------------------------------------------------
        # NORMALIZE LLM ANSWERS
        # ---------------------------------------------------------

        validation_df[llm_col] = (
            validation_df[llm_col]
            .astype(str)
            .str.strip()
        )

        # ---------------------------------------------------------
        # LLM CORRECTNESS
        # ---------------------------------------------------------

        validation_df['llm_correctness'] = np.where(

            validation_df[llm_col]
            ==
            validation_df['actual_match'],

            "LLM_CORRECT",

            "LLM_INCORRECT"
        )

        print("\nLLM correctness calculation completed")

        # ---------------------------------------------------------
        # OPTIONAL DETAILED STATUS
        # ---------------------------------------------------------

        validation_df['detailed_validation'] = np.select(

            [

                (
                    (validation_df['actual_match'] == "Yes")
                    &
                    (validation_df[llm_col] == "Yes")
                ),

                (
                    (validation_df['actual_match'] == "No")
                    &
                    (validation_df[llm_col] == "No")
                ),

                (
                    (validation_df['actual_match'] == "Yes")
                    &
                    (validation_df[llm_col] == "No")
                ),

                (
                    (validation_df['actual_match'] == "No")
                    &
                    (validation_df[llm_col] == "Yes")
                )

            ],

            [

                "Correct Match Detection",

                "Correct Mismatch Detection",

                "LLM Missed Correct Match",

                "LLM False Positive"

            ],

            default="Unable To Validate"
        )

        # ---------------------------------------------------------
        # FINAL OUTPUT COLUMNS
        # ---------------------------------------------------------

        final_cols = [

            'callid',

            'account_number',

            'Level 2',

            'CreditLimit',

            'transcript_credit_limit',

            'db_credit_limit',

            'actual_match',

            llm_col,

            'llm_correctness',

            'detailed_validation',

            'transcript'
        ]

        # keep only available cols
        final_cols = [
            c for c in final_cols
            if c in validation_df.columns
        ]

        validation_output = validation_df[final_cols]

        # ---------------------------------------------------------
        # PRINT SUMMARY
        # ---------------------------------------------------------

        print("\n================ SUMMARY ================\n")

        print(
            validation_output['llm_correctness']
            .value_counts(dropna=False)
        )

        print("\nDetailed validation:\n")

        print(
            validation_output['detailed_validation']
            .value_counts(dropna=False)
        )

        # ---------------------------------------------------------
        # EXPORT FILE
        # ---------------------------------------------------------

        output_file_name = (
            "llm_credit_limit_validation.xlsx"
        )

        validation_output.to_excel(
            output_file_name,
            index=False
        )

        print(f"\nValidation file saved: {output_file_name}")

        print("\n================ VALIDATION END ================\n")
        # df_ai_pandas['Output_final1'] = df_ai_pandas['ai_response3'].apply(safe_repair_and_load)
        # df_ai_pandas = pd.concat([df_ai_pandas,df_ai_pandas['Output_final1'].apply(pd.Series)], axis =1) 	

        df_ai_pandas.to_excel("InformationStatementIntermediateResults29jul.xlsx")
        print("Scoring done")

        #Final Scoring
        ai_response_column_list = [
        "Does the call involve Account Balance Request?",
        "Did the agent give all the asked details correctly as per database?",
        "Did the agent give information about current balance of customer/ billed balance on account to the customer?",
        "Did the agent give information about unbilled goods amount/good on approval to the customer?",
        "Did the agent give information about total amount/total commitment amount to the customer?",
        "Did the agent give information about the credit limit to the customer?",
        "Did the agent give information about the remaining credit to the customer?",
        "Did the agent give information about required payment amount/minimum monthly payment to the customer?",
        "Did the agent give information about when the payment is due by to the customer?",
        "Was the credit limit provided by the agent correct as per database?",
        "Was the Last Statement date provided by the agent correct as per database?",
        "Was the Last Statement number provided by the agent correct as per database?",
        "Was the Next Statement Date provided by the agent correct as per database?",
        "Was the Next Statement Number provided by the agent correct as per database?",
        "Was the Last Payment Amount provided by the agent correct as per database?",
        "Was the Last Payment Date provided by the agent correct as per database or equal Call date?",
        "Was the Last Order Date provided by the agent correct as per database or equal Call date?",
        "Was the Required Payment Due Date provided by the agent correct as per database?",
        "Evidence",
        "Evidence Account Info",
        "Is the call about Placing an order/involve delivery charge?",
        "Did the agent inform the Delivery charge Parcel Shop as 3.50?",
        "Did the agent inform the Delivery charge for Home delivery as 4.95?",
        "Did the agent inform that delivery status tracking email will be sent for home delivery?",
        "Did the agent inform that email/text message will be sent when item can be collected for store delivery?",
        
        "Did the agent inform the customer to carry their ID with them when collecting order from the store?",
        "Is the call about arranging an order collection?",
        "Did the agent inform the Collection charge Parcel Shop as 2.50?",
        "Did the agent inform the Collection charge for Home delivery as 2.50?",
        "Did the agent inform there is no Collection charge for store collection?",
        "Does the conversation involve caller having more than one NEXT account?",
        "Did the agent say that the customer can have a cash as well as credit account?",
        "Did the agent say that the customer can have two credit accounts with NEXT?",
        "Does the conversation involve caller having more than one account?",
        "Does the call involve collection of order from store?",
        "Did the caller ask what is the collection charge?",
        "Did the caller ask what is the delivery charge?",
        "Is the caller making a payment in the call?",
        "Was the payment successful?",
        "Did advisor mention that a confirmation of payment will be sent to registered email id/address?",
        "Did the agent specifically inform the caller that return charge is applicable for collection arranged?",
        "Was it a case of faulty item or wrong goods sent?",
        "Did the customer/caller explicitly ask for account balance?",
        "Did the customer/caller wanted to know their balance?",
        "Did the customer/caller ask about the statement date clearly/explicitly?",
        "Did the caller wanted to know about the statemet date?",
        "Was it mentioned in the transcript explicitly that payment was successful?",
        "Did the caller specifically ask the agent the charge on home delivery collection?",
        "Did the caller specifically ask the agent the charge on parcel shop collection?",
        "Did the transcript mention refund of charges?",
        "Did the caller specifically ask the agent the charge on home delivery?",
        "Did the caller specifically ask the agent the charge on parcel shop delivery?",
        "Does the call explicitly involve home delivery of products?",
        "Does the agent confidently/clearly mention the required details asked by caller?",
        "Was a store pickup booked during the call?",
        "Did the agent explanation resolve the doubts for caller?",
        "Is the caller confused/did not understand/not agree on the balance explanation?",
        "Did the agent say on transcript the collection charges will be refunded?",
        "Was the collection charges will be refunded promised in the trancript?",
        "Did the agent say email confirmation will be sent for refund of charges?"
    ]
        for col in ai_response_column_list:
            if col not in df_ai_pandas.columns:
                df_ai_pandas[col] = "N.A." 

        

        df_ai_pandas["iAuidt Result: Infromation Statement Explanation"]="N.A."
        df_ai_pandas['iAuidt Score: Infromation Statement Explanation'] = 10
        df_ai_pandas['iAudit Remarks: Infromation Statement Explanation']= ""



        #Account Balance Enquiry

        #Did the customer/caller ask about the statement date clearly/explicitly? Did the caller wanted to know about the statemet date?
        df_ai_pandas['Result: Statement Date'] = np.where(
            (df_ai_pandas["Does the call involve Account Balance Request?"].str.lower()=='yes')&
            (df_ai_pandas["Did the customer/caller ask about the statement date clearly/explicitly?"].str.lower()=='yes') &
            (df_ai_pandas["Does the agent confidently/clearly mention the required details asked by caller?"].str.lower()=="no") &
             (df_ai_pandas["Did the caller wanted to know about the statemet date?"].str.lower()=='yes') &
            ((df_ai_pandas["Was the Last Statement date provided by the agent correct as per database?"].str.lower()=='no') &
            (df_ai_pandas["Was the Next Statement Date provided by the agent correct as per database?"].str.lower()=='no')
            ),'Amber Error','No Error')

        # "Did the customer/caller explicitly ask for account balance?",
        #"Did the customer/caller wanted to know their balance?"
        #Is the caller confused/did not understand/not agree on the balance explanation?
        #or and logic
        df_ai_pandas['Result: Account Balance Explanation'] = np.where(

        (df_ai_pandas["Does the call involve Account Balance Request?"].str.lower()=='yes')& (df_ai_pandas["Did the customer/caller explicitly ask for account balance?"].str.lower()=='yes') &(df_ai_pandas["Did the customer/caller wanted to know their balance?"].str.lower()=='yes') & 
        (
        (
            (df_ai_pandas["Does the agent confidently/clearly mention the required details asked by caller?"].str.lower()=="no") | (df_ai_pandas["Did the agent explanation resolve the doubts for caller?"].str.lower()=='no') | (df_ai_pandas["Is the caller confused/did not understand/not agree on the balance explanation?"].str.lower()=="yes")
        )      |

        (
            (df_ai_pandas[ "Did the agent give information about current balance of customer/ billed balance on account to the customer?"].str.lower()=='no')&
            (df_ai_pandas["Did the agent give information about unbilled goods amount/good on approval to the customer?"].str.lower()=='no')&
            (df_ai_pandas["Did the agent give information about total amount/total commitment amount to the customer?"].str.lower()=='no' )&
            (df_ai_pandas["Did the agent give information about the credit limit to the customer?"].str.lower()=='no' )&
            (df_ai_pandas["Did the agent give information about the remaining credit to the customer?"].str.lower()=='no' )&
        (df_ai_pandas["Did the agent give information about required payment amount/minimum monthly payment to the customer?"].str.lower()=='no') 
        &
        (df_ai_pandas["Did the agent give information about when the payment is due by to the customer?"].str.lower()=='no') &
        (df_ai_pandas["Did the agent give all the asked details correctly as per database?"].str.lower()=='no')  
        )), 'Compliant with development', 'No Error')

        #Place an order Enquiry
        #  Did the caller specifically ask the agent the charge on home delivery?",
        # "Did the caller specifically ask the agent the charge on parcel shop delivery?"

        df_ai_pandas['Result: Delivery Charge'] = np.where( (df_ai_pandas["Is the call about Placing an order/involve delivery charge?"].str.lower()=='yes')&(df_ai_pandas["Did the caller ask what is the delivery charge?"].str.lower()=='yes') & (df_ai_pandas["Did the transcript mention refund of charges?"].str.lower()=='no') 
                &
                    (
                    ((df_ai_pandas["Did the caller specifically ask the agent the charge on parcel shop delivery?"].str.lower()=='yes')&(df_ai_pandas["Did the agent inform the Delivery charge Parcel Shop as 3.50?"].str.lower()=='no')) &
                    ((df_ai_pandas["Did the caller specifically ask the agent the charge on home delivery?"].str.lower()=='yes')&(df_ai_pandas["Did the agent inform the Delivery charge for Home delivery as 4.95?"].str.lower()=='no'))
                    
                    ) , 'Red Error', 'No Error'                   

        )

        # df_ai_pandas['Result: Delivery Status'] = np.where( 
        #             ((df_ai_pandas["Is the call about Placing an order/involve delivery charge?"].str.lower()=='yes') &
                    
        #             (
        #             ((df_ai_pandas["Does the call explicitly involve home delivery of products?"].str.lower()=='yes')& df_ai_pandas["Did the agent inform that delivery status tracking email will be sent for home delivery?"].str.lower()=='no') 
        #             &(
        #              (df_ai_pandas["Does the call involve collection of order from store?"].str.lower()=='yes') &
        #             (df_ai_pandas["Did the agent inform that email/text message will be sent when item can be collected for store delivery?"].str.lower()=='no'))
        #             &(df_ai_pandas["Did the transcript mention refund of charges?"].str.lower()=='no')
                    
        #             ) ), 'Compliant with development', 'No Error'                   

        # )
        #"Was a store pickup booked during the call?"
        df_ai_pandas['Result: Store Pickup'] = np.where( 
                    ((df_ai_pandas["Does the call involve collection of order from store?"].str.lower()=='yes') & (df_ai_pandas['Queues'].isin(['Voice Credit Investigate'])) &
                     (df_ai_pandas["Was a store pickup booked during the call?"].str.lower()=='yes') &
                      (df_ai_pandas["Did the transcript mention refund of charges?"].str.lower()=='no') &

                    (df_ai_pandas["Did the agent inform the customer to carry their ID with them when collecting order from the store?"].str.lower()=='no')
                    
                    ) , 'Compliant with development', 'No Error'                   

        )

        #Make Payment
        #Was it mentioned in the transcript explicitly that payment was successful?
        df_ai_pandas['Result: Successful Make Payment Email'] = np.where(
             ( 

            (df_ai_pandas["Is the caller making a payment in the call?"].str.lower()=='yes') 
            & (df_ai_pandas["Was the payment successful?"].str.lower()=='yes') 
            & (df_ai_pandas["Was it mentioned in the transcript explicitly that payment was successful?"].str.lower()=='yes')
            & (df_ai_pandas["Did advisor mention that a confirmation of payment will be sent to registered email id/address?"].str.lower()=='no')
            &(df_ai_pandas["Did the transcript mention refund of charges?"].str.lower()=='no')
                      )
            , 'Compliant with development' , 'No Error')





        #Collection Enquiry

        # "Did the caller specifically ask the agent the charge on home delivery collection?",
        # "Did the caller specifically ask the agent the charge on parcel shop collection?",
        # "Did the transcript mention refund of charges?"
        # "Did the agent say on transcript the collection charges will be refunded?",
        # "Was the collection charges will be refunded promised in the trancript?",
        # "Did the agent say email confirmation will be sent for refund of charges?"

        df_ai_pandas['Result: Collection Charge'] = np.where( (df_ai_pandas["Is the call about arranging an order collection?"].str.lower()=='yes')&(df_ai_pandas["Did the caller ask what is the collection charge?"].str.lower()=='yes') & (df_ai_pandas["Did the transcript mention refund of charges?"].str.lower()=='no') 
                    &(
                    ((df_ai_pandas["Did the caller specifically ask the agent the charge on parcel shop collection?"].str.lower()=='yes')&(df_ai_pandas["Did the agent inform the Collection charge Parcel Shop as 2.50?"].str.lower()=='no')) &
                    ((df_ai_pandas["Did the caller specifically ask the agent the charge on home delivery collection?"].str.lower()=='yes')&(df_ai_pandas["Did the agent inform the Collection charge for Home delivery as 2.50?"].str.lower()=='no'))
                    
                    ) , 'Red Error', 'No Error'                   

        )

        df_ai_pandas['Result: Collection Charges Refund Email'] = np.where( (df_ai_pandas["Did the caller ask what is the collection charge?"].str.lower()=='yes') & (df_ai_pandas["Did the transcript mention refund of charges?"].str.lower()=='yes') & (df_ai_pandas["Did the agent say on transcript the collection charges will be refunded?"].str.lower()=='yes')
         & (df_ai_pandas["Was the collection charges will be refunded promised in the trancript?"].str.lower()=='yes') 
         & (df_ai_pandas["Did the agent say email confirmation will be sent for refund of charges?"].str.lower()=='no')

            , "Compliant with development", "No Error"
        )




        df_ai_pandas['Result: Return Charge Applicable'] = np.where( 
                                                                    (

        (df_ai_pandas[ "Is the call about arranging an order collection?"].str.lower()=="yes")

        &(df_ai_pandas["Did the transcript mention refund of charges?"].str.lower()=='no')

        & (df_ai_pandas["Was it a case of faulty item or wrong goods sent?"].str.lower()=="no")

        & (df_ai_pandas[ "Did the agent specifically inform the caller that return charge is applicable for collection arranged?"].str.lower()=="no") 
        
        
        ), "Red Error", "No Error"

        )



        #Account Information Enquiry
        # df_ai_pandas['Result: Multiple Account Type'] = np.where( (df_ai_pandas["Does the conversation involve caller having more than one NEXT account?"].str.lower()=='yes')&
        #             (
        #             (df_ai_pandas["Did the agent say that the customer can have a cash as well as credit account?"].str.lower()=='yes') 
                    
        #             ) , 'Compliant with development', 'No Error'                   

        # )

        # df_ai_pandas['Result: Multiple Credit Account'] = np.where( (df_ai_pandas["Does the conversation involve caller having more than one NEXT account?"].str.lower()=='yes')&
        #             (
        #             (df_ai_pandas["Did the agent say that the customer can have two credit accounts with NEXT?"].str.lower()=='yes') 
                    
        #             ) , 'Amber Error', 'No Error'                   

        # )




        final_result_cols = [col for col in df_ai_pandas.columns if col.startswith("Result:")]
        df_ai_pandas['iAuidt Result: Infromation Statement Explanation']= df_ai_pandas[final_result_cols].apply(get_result_priority, axis = 1)
        # df_ai_pandas['iAuidt Result: Infromation Statement Explanation']= np.where((df_ai_pandas['Level 2'].isin(level_2_req_list) & (df_ai_pandas["Does the conversation involve caller having more than one NEXT account?"].str.lower()=='no')), df_ai_pandas['iAuidt Result: Infromation Statement Explanation'], 'N.A.' )
        # df_ai_pandas['iAuidt Result: Infromation Statement Explanation']=np.where(((df_ai_pandas["agent_order_max"]== df_ai_pandas["agent_order"])& (df_ai_pandas["Does the conversation involve caller having more than one NEXT account?"].str.lower()=='no') ),df_ai_pandas['iAuidt Result: Infromation Statement Explanation'], 'N.A.' )
        df_ai_pandas['iAuidt Result: Infromation Statement Explanation']=np.where(
                                                ((df_ai_pandas["Output_final"]=="") |(df_ai_pandas["Output_final"].isna()))
                                                , 'N.A.',df_ai_pandas['iAuidt Result: Infromation Statement Explanation'] 

        )
        result_cols = [col for col in df_ai_pandas.columns if col.startswith("Result:") ] 
        error_list = ["red error", "amber error", "compliant with development"]
        df_ai_pandas['Remarks'] = df_ai_pandas.apply(lambda row: ', '.join([f"{str(row[col]).strip()}: {col.replace('Result:', '').strip()}" for col in result_cols if str(row[col]).strip().lower() in error_list])
            , axis =1)

        df_ai_pandas['iAuidt Score: Infromation Statement Explanation']= df_ai_pandas['iAuidt Result: Infromation Statement Explanation'].map(score_map)	
        df_ai_pandas['iAudit Remarks: Infromation Statement Explanation']=	df_ai_pandas['Remarks'] 
        remarks_dict = {
        "Statement Date": "Correct Statement dates not shared by agent",
        "Account Balance Explanation": "Account Balance was not properly explained",
        "Delivery Charge": "Correct delivery charges not mentioned",
        "Delivery Status":"Delivery status tracking for home delivery not mentioned/text message for store delivery not mentioned",
        "Store Pickup": "Agent did not inform the customer to carry their ID with them when collecting order from the store",
        "Collection Charge": "Correct collection charges not mentioned",
        "Multiple Account Type": "Agent mentioned customer can have cash as well as credit account",
        "Multiple Credit Account": "Agent mentioned customer can have two credit accounts",
        "Successful Make Payment Email": "Agent did not inform that a confirmation of payment will be sent to registered email id/address",
        "Return Charge Applicable": "Agent did not inform the customer that return charge is applicable for collection arranged",
        "Collection Charges Refund Email": "Agent did not say email confirmation will be sent for refund of collection charges."
        }
        df_ai_pandas['iAudit Remarks: Infromation Statement Explanation'] = df_ai_pandas['iAudit Remarks: Infromation Statement Explanation'].replace(remarks_dict, regex = True)

        df_ai_pandas['iAudit Remarks: Infromation Statement Explanation'] = np.where(df_ai_pandas['iAuidt Score: Infromation Statement Explanation']==10,"",df_ai_pandas['iAudit Remarks: Infromation Statement Explanation'] )



# ##################################################################################################################
        # Preserve NA info in remarks/comments
    # Preserve NA info in remarks/comments
        df_ai_pandas['iAudit Remarks: Infromation Statement Explanation'] = np.where(
            df_ai_pandas['iAuidt Result: Infromation Statement Explanation'] == 'N.A.',
            'N.A.',
            df_ai_pandas['iAudit Remarks: Infromation Statement Explanation']
        )

        # Convert NA result to Good Customer Outcome
        df_ai_pandas['iAuidt Result: Infromation Statement Explanation'] = np.where(
            df_ai_pandas['iAuidt Result: Infromation Statement Explanation'] == 'N.A.',
            'Good Customer Outcome',
            df_ai_pandas['iAuidt Result: Infromation Statement Explanation']
        )


        df_ai_pandas.to_excel("InformationStatementExplanationIntermediateAfterChange29jul.xlsx")

        # Add Confidence column if not present from LLM output
        if 'Confidence' not in df_ai_pandas.columns:
            df_ai_pandas['Confidence'] = ''

        # Define intermediate output columns
        intermediate_cols = [
            'createdate',        # Call date
            'callid',            # Call id
            'ticket_id',         # Ticket id
            'DepartmentName',    # Department
            'CountryName',       # Region
            'AgentId',           # Agent id
            'assignee',          # Agent name
            'Level 2',           # Level 2/work group
            'Channel_logger',    # Call type
            'iAuidt Result: Infromation Statement Explanation',
            'iAuidt Score: Infromation Statement Explanation',
            'iAudit Remarks: Infromation Statement Explanation',
            'Confidence'
        ]
        # Keep only columns that exist in the dataframe
        intermediate_cols = [c for c in intermediate_cols if c in df_ai_pandas.columns]
        df_ai_pandas_filtered = df_ai_pandas[intermediate_cols].copy()

        # Rename columns to user-friendly names
        df_ai_pandas_filtered = df_ai_pandas_filtered.rename(columns={
            'createdate': 'Call date',
            'callid': 'Call id',
            'ticket_id': 'Ticket id',
            'DepartmentName': 'Department',
            'CountryName': 'Region',
            'AgentId': 'Agent id',
            'assignee': 'Agent name',
            'Level 2': 'Level 2/Work group',
            'Channel_logger': 'Call type'
        })

        replace_map_result ={ 'Amber Error': 'Non-Compliant No Poor Outcome',
                      'Red Error': 'Non-Compliant Poor Outcome',
                      'Compliant with development': 'Compliant with Development',
                      'Good Customer Outcome': 'Compliant - Good outcome',
                      'N.A.': 'N.A.'
                      
        }
        df_ai_pandas['iAuidt Result: Infromation Statement Explanation']= np.where(df_ai_pandas['Level 2'].isin(level_2_req_list), df_ai_pandas['iAuidt Result: Infromation Statement Explanation'], 'N.A.')
        df_ai_pandas['iAuidt Result: Infromation Statement Explanation']=np.where(df_ai_pandas["agent_order_max"]== df_ai_pandas["agent_order"],df_ai_pandas['iAuidt Result: Infromation Statement Explanation'], 'N.A.')

        df_ai_pandas_filtered['iAuidt Result: Infromation Statement Explanation'] = df_ai_pandas_filtered['iAuidt Result: Infromation Statement Explanation'].replace(replace_map_result).replace(['', None, np.nan], 'N.A.')
        df_ai_pandas['iAuidt Score: Infromation Statement Explanation']= np.where(
            df_ai_pandas_filtered['iAuidt Result: Infromation Statement Explanation'].str.contains('N.A.'),10, df_ai_pandas_filtered['iAuidt Score: Infromation Statement Explanation']
            )

        print(df_ai_pandas.columns.tolist())       
        print(
            df_ai_pandas[
                [
                    "Level 2",
                    "iAuidt Result: Infromation Statement Explanation",
                    "iAuidt Score: Infromation Statement Explanation"
                ]
            ].head(50)
        )

        print(
            df_ai_pandas[
                'iAuidt Result: Infromation Statement Explanation'
            ].value_counts(dropna=False)
        )

        df_ai_pandas_filtered.to_excel(result_path)
    except Exception as e:
        log.error(f"Module name: Information Statement Explanation- Error: {e}", exc_info=True)
            
   





















    
    
    