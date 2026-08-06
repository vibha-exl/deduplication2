# !pip install  json-repair  -q

import warnings
import os,sys
import json
from json_repair import repair_json
import re
from datetime import datetime
import numpy as np
import pandas as pd
from pyspark.errors import PySparkException
from pyspark.sql.types import StructType, StructField, BooleanType, StringType
from pyspark.sql.functions import from_json, col, lit, udf, when, expr
# sys.path.insert(0, '/Workspace/root/ContactCentre/iAudit Processes/deployment2')
# sys.path.insert(0, '/Workspace/root/ContactCentre/iAudit Processes/deployment2/batch3')
sys.path.insert(0, '/Workspace/root/ContactCentre/iAudit Processes/deployment2/iAudit_vdev/')
from utils.iaudit_logger import get_logger
from prompts.InformationStatementExplanationPrompts import *
from pyspark.sql.functions import format_string
from pyspark.sql.functions import array, lit, when, size, concat_ws
from pyspark.sql import functions as F
import yaml

from dev_helpers import export_to_excel

warnings.filterwarnings("ignore")

# transcript_path = r'/Workspace/Users/ananthagiri_abhiram@next.co.uk/iAudit_Trisha/AllAuditedTransciptNonRedacted2026.xlsx'

# #one day call
# transcript_path = r'/Workspace/Users/ananthagiri_abhiram@next.co.uk/iAudit_Trisha/combined_file_2026_03_27.xlsx'

# result_path = r"InformationExplanationExplanationFinalResults27032026V2.xlsx"
log = get_logger()

def safe_repair_and_load(s):
    try:
        repaired = repair_json(s)
        return json.loads(repaired)
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

########################################################################################################################################################
PAYMENT_SUCCESS_KEYWORDS = [

    "payment has been success",
    "payment has been processed",
    "payment has been accepted",
    "payment has been completed",
    "payment success",
    "payment got processed",
    "payment is processed",
    
    "processed the payment",
    "payment has been taken successfully",
    "payment has been accepted successfully",
    "payment has been accepted",
    "payment successful",
    "payment was successful",
    "payment is successful",
    "payment has gone through",
    "gone through successfully",
    "payment for £100 is successful",
    "payment has been successfully accepted",
    "payment will show on your account tomorrow",
    "payment will reflect on your account"
]

def contains_payment_success_keywords(text):
    if pd.isna(text):
        return False

    text = str(text).lower()

    return any(
        keyword.lower() in text
        for keyword in PAYMENT_SUCCESS_KEYWORDS
    )

if __name__=="__main__":
    try:
        main_config = yaml.safe_load(open('../main_config.yaml', 'r'))
        catalog_config = yaml.safe_load(open('../catalog_config.yaml', 'r'))
        llama_8b = main_config['LLM']['llama']
        llama_70b = main_config['LLM']['llama_large']

        formatted_date = '2026-07-27_testing'
        # try:
        #     formatted_date = dbutils.jobs.taskValues.get(taskKey="Pre-Modules-Run", key="sql_date")
        # except Exception as e:
        #     formatted_date = sys.argv[1]
        #     if not re.match(r"^\d{4}-\d{2}-\d{2}$", str(formatted_date)):
        #         raise ValueError(f"formatted_date '{formatted_date}' is not in yyyy-mm-dd format")
        #     logger.info(f"formatted_date: {formatted_date}")

        # INPUT_FILE_PATH = spark.table(catalog_config['output_table']['filtered_table']).filter(f"call_date = '{formatted_date}'")

        call_ids = [
            "0cca37d9-8710-42e2-8080-ae0a1c69c829", "3716c8a6-218a-4e82-8d21-5364dfe60516", "09c64c0d-4ff2-448d-99a9-9c644702b3b3", "3147d444-5db5-4af9-84c9-5ed6335487b2", "019446d9-68d0-458b-9ae8-843da8a56ebd", "04598191-bc6c-4e6c-be23-0a61795d8894", "253ad0d7-0d5a-4a47-b596-2c159198f311", "382abd18-d9e0-4eea-bcab-55ee0b42a0cc", "2001f0d6-1902-435c-8388-10ba29ad3f49", "0f4ea907-54c0-4993-b42d-4d67e5450786", "1840b86d-4429-4ea7-95cf-95c782feec0c", "0fcb8d2f-6c72-4a07-add2-fc12c31051ad", "68161c07-0b2b-4944-8af9-d3d8b724e158", "5c9b8b49-e20c-4ef7-bd82-1776049f0447", "71d4deae-c576-4019-a4ef-bd941eb1010a", "579060ba-3623-4d8a-b653-e5ddea0b54c9", "9ef81f1f-9952-4122-89f5-8ee40d6fe37a", "9f1bf986-799f-42ea-ae86-02c31fb7cf07", "a2bab2fb-7755-490c-ac21-8b98feba7880", "a82fc757-1e62-44ce-91d0-67025c66695c", "cdd13a55-7a9d-4ac4-a0db-41626c46bfff", "c9f19c28-c6f3-488f-a1cf-7823d6db7ae5", "ef90635d-f392-47af-a42d-f0d03792053d", "d933aed3-c031-4f53-b9f0-021a58bf3da4", "d62fba26-3a4b-4081-876f-8fa0a9320235", "f227b314-91c5-437a-81b9-7d6d694c4820", "d539600b-2b1e-457e-ae1c-cff833167cb8", "dfbc534d-8924-4f7a-8e14-12611e102a7f", "de73f101-ce1f-4516-b340-e05bb7ce0029", "e329baa4-4a53-4451-ba53-994b908f30d9", "cdff6c72-c661-406b-a910-5de094186156", "e6a0edeb-6cca-4812-89c0-64096ba89c8e", "c938e196-239f-4a22-b382-ed2bdae7f230"
        ]
        call_ids_str = ",".join([f"'{cid}'" for cid in call_ids])
        INPUT_FILE_PATH = spark.sql(f"""
            SELECT *
            FROM {catalog_config['output_table']['filtered_table']}
            WHERE callid IN ({call_ids_str})
        """)

        transcripts =  INPUT_FILE_PATH.toPandas()
        transcripts = transcripts.rename(columns={'AgentID':'AgentId'})

        total_calls = len(transcripts['callid'].unique().tolist())
        filtered_calls = transcripts['callid'].unique().tolist()[:total_calls]
        # filtered_calls = transcripts['callid'].unique().tolist()[:100]
        df_transcripts = transcripts[transcripts['callid'].isin(filtered_calls)].reset_index(drop=True)
    
        audit_callids = df_transcripts['callid'].unique().tolist()

        # df_transcripts['createdate'] = pd.to_datetime(df_transcripts['createdate'], format = 'mixed', errors='coerce')
        # df_transcripts = df_transcripts.sort_values(by=["callid", 'createdate'])
        # df_transcripts['createdate'] = df_transcripts['createdate'].astype(str)

        spark_logger =  spark.sql(f"""
                SELECT `WorkGroupName`, `conversation_id`, `ResponseName`, `account_number`, `call_date`, `Supplier` FROM contactcentre_prod.staging.zen_live9 
                WHERE conversation_id IN {tuple(audit_callids)}      
            """)

        df_logger = spark_logger.toPandas()
        df_logger['call_date'] = pd.to_datetime(df_logger['call_date']).dt.date

        d1 = {
            'WorkGroupName': 'Queues',
            'conversation_id': 'CallId',
            'ResponseName': 'Level 2',
        }

        rename_cols = {k:v for k,v in d1.items() if v!=''}
        df_logger = df_logger.rename(columns=rename_cols)
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
        # print("with customer data",df_logger.columns)
        # print("with customer data",df_logger.shape)
        # print("customer data",df_vw_customers.shape)
        # print("customer data unique account_number",len(df_vw_customers['account_number'].unique().tolist()))

        ############################## GOGW Changes ##############################

        gogw_df = pd.read_excel(
            main_config['File_Paths']['adjustment_path'],
            sheet_name=main_config['adjustment_sheet']
            # sheet_name="Adjusments logged Jun-26"
        )

        gogw_df.columns = gogw_df.columns.str.strip()
        gogw_df['Account Number'] = gogw_df['Account Number'].astype(str).str.strip()
        gogw_df['DateStamp'] = pd.to_datetime(gogw_df['DateStamp']).dt.date
        gogw_df['Adj Amount'] = pd.to_numeric(gogw_df['Adj Amount'], errors='coerce')
        gogw_df['DateStamp'] = pd.to_datetime(gogw_df['DateStamp'],errors='coerce').dt.date

        df_logger['account_number'] = df_logger['account_number'].astype(str).str.strip()
        df_logger['call_date'] = pd.to_datetime(df_logger['call_date']).dt.date

        # Remove invalid rows
        gogw_df = gogw_df[
            gogw_df['Account Number'].notna()
            & (gogw_df['Account Number'].str.strip() != "")
            & gogw_df['DateStamp'].notna()
        ]

        # Keep only positive adjustments
        gogw_df = gogw_df[gogw_df['Adj Amount'].fillna(0) > 0]

        # One record per Account Number + Date
        gogw_df = (
            gogw_df
            .groupby(['Account Number', 'DateStamp'], as_index=False)
            .agg({ 'Adj Amount': 'sum' })
        )

        # Create merge key
        gogw_df['merge_key'] = (
            gogw_df['Account Number'] + "_" + gogw_df['DateStamp'].astype(str)
        )

        df_logger['merge_key'] = (
            df_logger['account_number'].astype(str).str.strip()
            + "_" + df_logger['call_date'].astype(str)
        )


        gogw_merge_keys = set(gogw_df['merge_key'])

        df_logger['refund_found'] = np.where(
            df_logger['merge_key'].isin(gogw_merge_keys),
            1,
            0
        )
        
        
        REFUND_INITIATED_COL = "Was the refund initiated by the agent during this conversation?"
        REFUND_NOTIFICATION_COL = "Did the agent inform the customer that an email confirmation will be sent for the refund?"
        REFUND_AMOUNT_RECONFIRMATION_COL = "Did the agent reconfirm the refund amount to the customer?"

        ############################## GOGW Changes ##############################

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
        df_logger['priority'] =df_logger['Level 2'].isin(level_2_req_list).astype(int)
        # print("Logger before",df_logger.shape)
        df_logger_unique = (df_logger.sort_values(['CallId', 'priority'], ascending=[True, False]).drop_duplicates(subset = 'CallId', keep='first').drop(columns='priority'))
        # print(df_logger_unique.shape)

        data = pd.merge(df_transcripts, df_logger_unique, left_on='callid', right_on='CallId', how='left')
        data['callid_agentid']= data['callid'] + "|" + data['AgentId']
        print("data",data.columns)
        grouped_transcript_all = data.groupby('callid_agentid').apply(lambda x: "\n".join(f"{row['channel']}: {row['transcript']}" for _, row in x.iterrows())).reset_index(name = 'transcript')
        data_intermediate = data.groupby('callid_agentid', as_index = False).first()
        data_req = pd.merge(grouped_transcript_all, data_intermediate, on='callid_agentid', how='left')
        
        data_req['call_date'] = data_req['call_date_x']

        # data_req['createdate'] = pd.to_datetime(data_req['createdate'], format = 'mixed', errors='coerce')
        data_req['createdate'] = pd.to_datetime(data_req['createdate'], utc=True, errors='coerce')
        data_req = data_req.sort_values(by=["callid", 'createdate'], ascending =[True, True])
        data_req['createdate'] = data_req['createdate'].astype(str)
        data_req['agent_order'] = data_req.groupby("callid").cumcount()+1
        data_req['agent_order_max'] = data_req.groupby("callid")["agent_order"].transform("max")

        data_req['StatementDate_y'] = pd.to_datetime(data_req['StatementDate_y'], errors='coerce')
        data_req['required_payment_due_date'] = data_req['StatementDate_y'] + pd.DateOffset(days= -7)
        print("final data",data_req.shape)

        #data_req.to_excel("check_data_information_statement.xlsx")

        # data_req = pd.read_excel(r"/Workspace/Users/ananthagiri_abhiram@next.co.uk/iAudit_Trisha/check_data_information_statement.xlsx")

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
        ########################################################################################################################################################

        data_req['payment_success_call'] = (
            (data_req['Level 2'] == 'Make Payment')
            &
            (data_req['transcript_x'].apply(contains_payment_success_keywords))
        )

        # payment_candidates = data_req[
        #     data_req['payment_success_call']
        # ].copy()

        # payment_candidates['payment_success'] = 'YES'
        # payment_candidates['email_present'] = ''

        # payment_candidates[
        #     [
        #         'callid_agentid',
        #         'transcript
        # ',
        #         'Level 2',
        #         'payment_success',
        #         'email_present'
        #     ]
        # ].to_excel(
        #     'payment_success_candidates_before_llm.xlsx',
        #     index=False
        # )


        data_req['payment_email_check_indicator'] = np.where(
            data_req['payment_success_call'],
            1,
            0
        )
########################################################################################################################################################


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
                                                            F.col("transcript_x"),
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
                                                            F.col("transcript_x")
                                                
                                                            
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
                                                            F.col("transcript_x")
                                                
                                        
                                                            )))
                            

            )
        
        # databricks-meta-llama-3-1-8b-instruct
        df_ai = (
            df_prompts.withColumn(
                "ai_response", F.when(
                    (F.col("Level 2").isin(level_2_req_list)) &
                (F.col("agent_order_max")== F.col("agent_order")) &
                (F.col("user_prompt") != ""), F.expr(
                    f"""ai_query('{llama_8b}',request => final_prompt)"""
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
        # df_ai_pandas.to_excel("InformationStatementExplanationv1.xlsx")
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
        # df_ai_pandas['Output_final1'] = df_ai_pandas['ai_response3'].apply(safe_repair_and_load)
        # df_ai_pandas = pd.concat([df_ai_pandas,df_ai_pandas['Output_final1'].apply(pd.Series)], axis =1) 	

        # df_ai_pandas.to_excel("InformationStatementIntermediateResults.xlsx")
        print("Scoring done")
        export_to_excel(
            df_ai_pandas,
            f"voice_information_statement_intermediate1_{formatted_date if 'formatted_date' in locals() else datetime.now().strftime('%Y-%m-%d')}",
            "voice",
            "InformationStatementExplanation"
        )

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
        "Did the agent say email confirmation will be sent for refund of charges?",
        "Did the agent inform the customer that an email confirmation will be sent for the refund?",
        "Was the refund initiated by the agent during this conversation?",
        "Did the agent reconfirm the refund amount to the customer?"
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

        # ========================================================================================
        # MAKE PAYMENT - Email/Notification Confirmation Check
        # ========================================================================================
        # FIRST PASS (from user_prompt_place_an_order):
        # The prompt includes an instruction note telling the LLM to answer "Yes" for the
        # payment confirmation question if the advisor mentioned ANY form of confirmation or
        # notification (e.g. "you'll get a notification shortly") - not just the word "email".
        # The JSON key remains unchanged for column name stability.
        #
        # If the first pass says "No" (agent didn't mention any confirmation/notification),
        # the result becomes 'Compliant with development' (i.e. a remark is flagged).
        # ========================================================================================
        df_ai_pandas["email_id_present_on_account"] = np.where(
            (
                (df_ai_pandas["Did the agent offer to update email for user account?"].str.lower() == "yes")
                & (df_ai_pandas["If agent offered did the user agree to update email on the account?"].str.lower() == "no")
            ),
            "no",
            "yes"
        )

        df_ai_pandas['Result: Successful Make Payment Email'] = np.where(
            (
                (df_ai_pandas["email_id_present_on_account"] == "yes")
                & (df_ai_pandas["Is the caller making a payment in the call?"].str.lower()=='yes') 
                & (df_ai_pandas["Was the payment successful?"].str.lower()=='yes') 
                & (df_ai_pandas["Was it mentioned in the transcript explicitly that payment was successful?"].str.lower()=='yes')
                & (df_ai_pandas["Did advisor mention that a confirmation of payment will be sent to registered email id/address?"].str.lower()=='no')
                &(df_ai_pandas["Did the transcript mention refund of charges?"].str.lower()=='no')
            ),
            'Compliant with development',
            'No Error'
        )
        
        # ========================================================================================
        # SECOND PASS - Only runs on flagged calls ('Compliant with development')
        # ========================================================================================
        # PURPOSE: When the first pass flags a call (agent didn't mention confirmation),
        # this second LLM check determines if the remark should be SUPPRESSED because:
        #   - The customer has no email on the account AND refused to add one when the agent
        #     offered. In this scenario, the agent cannot send a confirmation email, so the
        #     remark "Agent did not inform that a confirmation of payment will be sent" is
        #     not appropriate.
        #
        # WHY NOT IN FIRST PASS: The "customer refused email update" check requires a focused
        # prompt with specific instructions. The first pass is a broad multi-question prompt
        # that cannot reliably detect this nuanced conversational pattern.
        #
        # CONDITION: Only runs when payment_email_check_indicator==1 AND first pass flagged
        # a remark ('Compliant with development'). This keeps the second LLM call targeted
        # to only the calls that actually need re-evaluation.
        # ========================================================================================
        df_ai_spark = spark.createDataFrame(df_ai_pandas)
        prompt_email_confirmation_json = json.dumps(prompt_payment_email_confirmation)
        
        df_ai_spark = (
            df_ai_spark.withColumn(
                "ai_response_payment_email", F.when(
                    (F.col("email_id_present_on_account") == "yes") &
                    (F.col("payment_email_check_indicator") == 1) &
                    (F.col("Result: Successful Make Payment Email") == 'No Error'),
                    F.expr(
                        f"""ai_query('{llama_8b}', request => concat({prompt_email_confirmation_json}, transcript_x))"""
                    )
                )
            )
        )

        df_ai_pandas = df_ai_spark.toPandas()
        # df_ai_pandas.to_excel('../temp/inf_statements_inter2.xlsx')
        mask1 = (
            (df_ai_pandas["email_id_present_on_account"] == "yes") &
            (df_ai_pandas['payment_email_check_indicator'] == 1) &
            (df_ai_pandas['Result: Successful Make Payment Email'] == 'No Error')
        )
        df_ai_pandas.loc[mask1, 'ai_response_payment_email'] = (
            df_ai_pandas.loc[mask1, 'ai_response_payment_email'].apply(safe_repair_and_load)
        )

        # Extract the two answers from the second LLM response into separate columns
        df_ai_pandas = pd.concat(
            [
                df_ai_pandas,
                df_ai_pandas['ai_response_payment_email'].apply(pd.Series).rename(columns={
                    'Did advisor mention that a confirmation of payment will be sent to registered email id/address?': 'Did advisor mention that a confirmation of payment will be sent to registered email id/address_confirmation?'
                })
            ],
            axis=1
        )

        # Safety: ensure columns exist even if LLM did not return them for some rows
        if 'Did advisor mention that a confirmation of payment will be sent to registered email id/address_confirmation?' not in df_ai_pandas.columns:
            df_ai_pandas['Did advisor mention that a confirmation of payment will be sent to registered email id/address_confirmation?'] = "N.A."

        df_ai_pandas['Result: Successful Make Payment Email'] = np.where(
            df_ai_pandas['Did advisor mention that a confirmation of payment will be sent to registered email id/address_confirmation?'].str.lower() == 'no',
            'Compliant with development',
            df_ai_pandas['Result: Successful Make Payment Email']
        )

        #Collection Enquiry

        # "Did the caller specifically ask the agent the charge on home delivery collection?",
        # "Did the caller specifically ask the agent the charge on parcel shop collection?",
        # "Did the transcript mention refund of charges?"
        # "Did the agent say on transcript the collection charges will be refunded?",
        # "Was the collection charges will be refunded promised in the trancript?",
        # "Did the agent say email confirmation will be sent for refund of charges?"

        # Safety: re-initialize columns that may have been lost during the Spark roundtrip above
        # (spark.createDataFrame -> withColumn -> toPandas can drop columns with mixed/object types)
        for col in ai_response_column_list:
            if col not in df_ai_pandas.columns:
                df_ai_pandas[col] = "N.A."

        df_ai_pandas['Result: Collection Charge'] = np.where(
            (df_ai_pandas["Is the call about arranging an order collection?"].str.lower()=='yes')
            & (df_ai_pandas["Did the caller ask what is the collection charge?"].str.lower()=='yes')
            & (df_ai_pandas["Did the transcript mention refund of charges?"].str.lower()=='no') 
            & (
                (
                    (df_ai_pandas["Did the caller specifically ask the agent the charge on parcel shop collection?"].str.lower()=='yes')
                    & (df_ai_pandas["Did the agent inform the Collection charge Parcel Shop as 2.50?"].str.lower()=='no')
                ) 
                & (
                    (df_ai_pandas["Did the caller specifically ask the agent the charge on home delivery collection?"].str.lower()=='yes')
                    & (df_ai_pandas["Did the agent inform the Collection charge for Home delivery as 2.50?"].str.lower()=='no')
                )
            ), 
            'Red Error',
            'No Error'
        )

        df_ai_pandas['Result: Collection Charges Refund Email'] = np.where( (df_ai_pandas["Did the caller ask what is the collection charge?"].str.lower()=='yes') & (df_ai_pandas["Did the transcript mention refund of charges?"].str.lower()=='yes') & (df_ai_pandas["Did the agent say on transcript the collection charges will be refunded?"].str.lower()=='yes')
         & (df_ai_pandas["Was the collection charges will be refunded promised in the trancript?"].str.lower()=='yes') 
         & (df_ai_pandas["Did the agent say email confirmation will be sent for refund of charges?"].str.lower()=='no')

            , "Compliant with development", "No Error"
        )

        export_to_excel(
            df_ai_pandas,
            f"voice_information_statement_intermediate2_{formatted_date if 'formatted_date' in locals() else datetime.now().strftime('%Y-%m-%d')}",
            "voice",
            "InformationStatementExplanation"
        )
    
        ############################## GOGW Changes ##############################


        refund_cols = [REFUND_INITIATED_COL, REFUND_NOTIFICATION_COL, REFUND_AMOUNT_RECONFIRMATION_COL]
        refund_callids = data_req.loc[data_req["refund_found"] == 1, "callid"].unique()
        data_req_refund = data_req[
            data_req["callid"].isin(refund_callids)
            & (data_req["agent_order"] == data_req["agent_order_max"])
        ].copy()

        print(f"{len(refund_callids)=}")
        if len(refund_callids) > 0:
            prompt_refund_email_confirmation_json = json.dumps(prompt_refund_email_confirmation)

            df_refund_llm = (
                spark.createDataFrame(data_req_refund)
                .withColumn(
                    "ai_resp_refund_notif",
                    F.expr(f"""ai_query('{llama_8b}', request => concat({prompt_refund_email_confirmation_json}, transcript_x))""")
                )
                .toPandas()
            )

            # Parse refund notification email response
            df_refund_llm['_parsed_notif'] = df_refund_llm['ai_resp_refund_notif'].apply(safe_repair_and_load)
            df_notif_parsed = pd.concat(
                [
                    df_refund_llm[['callid']],
                    df_refund_llm['_parsed_notif'].apply(pd.Series)
                ],
                axis=1
            )

            for col in refund_cols:
                if col not in df_notif_parsed.columns:
                    df_notif_parsed[col] = "N.A."

            refund_notif_results = (
                df_notif_parsed[['callid', *refund_cols]]
                .drop_duplicates(subset=['callid'])  ##! Is this required?
            )

        else:
            refund_notif_results = pd.DataFrame(columns=['callid', *refund_cols])

        # Merge Refund Notification Email result; non-GOGW calls get N.A.
        # Drop refund_cols from df_ai_pandas before merge to avoid _x/_y suffix conflicts
        df_ai_pandas = df_ai_pandas.drop(columns=[c for c in refund_cols if c in df_ai_pandas.columns], errors='ignore')
        df_ai_pandas = df_ai_pandas.merge(refund_notif_results, on='callid', how='left')

        for col in refund_cols:
            if col not in df_ai_pandas.columns:
                df_ai_pandas[col] = "N.A."
            else:
                df_ai_pandas[col] = df_ai_pandas[col].fillna("N.A.")

        # mask = (
        #     (df_ai_pandas['refund_found'] == 1)
        #     &
        #     (df_ai_pandas[REFUND_INITIATED_COL].str.lower() == "yes")
        #     &
        #     (
        #         (df_ai_pandas[REFUND_NOTIFICATION_COL].str.lower() == "no")
        #         |
        #         (df_ai_pandas[REFUND_AMOUNT_RECONFIRMATION_COL].str.lower() == "no")
        #     )
        # )

        # df_ai_pandas.loc[
        #     mask,
        #     'Result: Refund Notification Email'
        # ] = "Compliant with development"

        
        df_ai_pandas['Result: Refund Notification Email'] = np.where(
            (
                (df_ai_pandas['refund_found'] == 1)
                & (df_ai_pandas[REFUND_INITIATED_COL].str.lower() == "yes")
                & (df_ai_pandas[REFUND_NOTIFICATION_COL].str.lower() == "no")
            ),
            'Compliant with development',
            'No Error'
        )
        
        df_ai_pandas['Result: Refund Amount Reconfirmation'] = np.where(
            (
                (df_ai_pandas['refund_found'] == 1)
                & (df_ai_pandas[REFUND_INITIATED_COL].str.lower() == "yes")
                & df_ai_pandas[REFUND_AMOUNT_RECONFIRMATION_COL].str.lower() == "no"
            ),
            'Compliant with development',
            'No Error'
        )

        ############################## GOGW Changes ##############################

        # Add condition where supplier is not_null in that case it's fine to not mention return charges.
        df_ai_pandas['Result: Return Charge Applicable'] = np.where( 
            (
                # df_ai_pandas["Supplier"].notna()
                # & (df_ai_pandas["Supplier"].astype(str).str.strip() != "") & 
                (df_ai_pandas[ "Is the call about arranging an order collection?"].str.lower()=="yes")
                &(df_ai_pandas["Did the transcript mention refund of charges?"].str.lower()=='no')
                & (df_ai_pandas["Was it a case of faulty item or wrong goods sent?"].str.lower()=="no")
                & (df_ai_pandas[ "Did the agent specifically inform the caller that return charge is applicable for collection arranged?"].str.lower()=="no")
            ),
            "Red Error",
            "No Error"
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


        export_to_excel(
            df_ai_pandas,
            f"voice_information_statement_intermediate4_{formatted_date if 'formatted_date' in locals() else datetime.now().strftime('%Y-%m-%d')}",
            "voice",
            "InformationStatementExplanation"
        )

        final_result_cols = [col for col in df_ai_pandas.columns if col.startswith("Result:")]
        # print(f"{final_result_cols=}")
        # print("Refund Notification Count: ", df_ai_pandas['Result: Refund Notification Email'].value_counts())
        # print("Refund Amount Reconfirmation Count: ", df_ai_pandas['Result: Refund Amount Reconfirmation'].value_counts())
        
        df_ai_pandas['iAuidt Result: Infromation Statement Explanation']= df_ai_pandas[final_result_cols].apply(get_result_priority, axis = 1)
        # df_ai_pandas['iAuidt Result: Infromation Statement Explanation']= np.where((df_ai_pandas['Level 2'].isin(level_2_req_list) & (df_ai_pandas["Does the conversation involve caller having more than one NEXT account?"].str.lower()=='no')), df_ai_pandas['iAuidt Result: Infromation Statement Explanation'], 'N.A.' )
        # df_ai_pandas['iAuidt Result: Infromation Statement Explanation']=np.where(((df_ai_pandas["agent_order_max"]== df_ai_pandas["agent_order"])& (df_ai_pandas["Does the conversation involve caller having more than one NEXT account?"].str.lower()=='no') ),df_ai_pandas['iAuidt Result: Infromation Statement Explanation'], 'N.A.' )
        df_ai_pandas['iAuidt Result: Infromation Statement Explanation']=np.where(
                                                ((df_ai_pandas["Output_final"]=="") |(df_ai_pandas["Output_final"].isna()))
                                                , 'N.A.',df_ai_pandas['iAuidt Result: Infromation Statement Explanation'] 

        )
        result_cols = [col for col in df_ai_pandas.columns if col.startswith("Result:") ] 
        error_list = ["red error", "amber error", "compliant with development"]
        df_ai_pandas['Remarks'] = df_ai_pandas.apply(
            lambda row: ', '.join([
                f"{str(row[col]).strip()}: {col.replace('Result:', '').strip()}"
                for col in result_cols
                if str(row[col]).strip().lower() in error_list
            ]),
            axis =1
        )

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
            "Collection Charges Refund Email": "Agent did not say email confirmation will be sent for refund of collection charges.",
            "Refund Notification Email": "Agent did not inform the customer that an email confirmation will be sent for the refund",
            "Refund Amount Reconfirmation": "Agent did not reconfirm the refund amount with the customer"
        }
        df_ai_pandas['iAudit Remarks: Infromation Statement Explanation'] = df_ai_pandas['iAudit Remarks: Infromation Statement Explanation'].replace(remarks_dict, regex = True)

        df_ai_pandas['iAudit Remarks: Infromation Statement Explanation'] = np.where(df_ai_pandas['iAuidt Score: Infromation Statement Explanation']==10,"",df_ai_pandas['iAudit Remarks: Infromation Statement Explanation'] )

        # df_ai_pandas.to_excel("InformationStatementExplanationIntermediateAfterChange.xlsx")
        df_ai_pandas_filtered = df_ai_pandas[['callid_agentid','callid','AgentId','iAuidt Result: Infromation Statement Explanation','iAuidt Score: Infromation Statement Explanation','iAudit Remarks: Infromation Statement Explanation']]

        replace_map_result ={ 'Amber Error': 'Non-Compliant No Poor Outcome',
                      'Red Error': 'Non-Compliant Poor Outcome',
                      'Compliant with development': 'Compliant with Development',
                      'Good Customer Outcome': 'Good Customer Outcome',
                      'N.A.': 'N.A.'
                      
        }
        df_ai_pandas['iAuidt Result: Infromation Statement Explanation']= np.where(df_ai_pandas['Level 2'].isin(level_2_req_list), df_ai_pandas['iAuidt Result: Infromation Statement Explanation'], 'N.A.')
        df_ai_pandas['iAuidt Result: Infromation Statement Explanation']=np.where(df_ai_pandas["agent_order_max"]== df_ai_pandas["agent_order"],df_ai_pandas['iAuidt Result: Infromation Statement Explanation'], 'N.A.')

        df_ai_pandas_filtered['iAuidt Result: Infromation Statement Explanation'] = df_ai_pandas_filtered['iAuidt Result: Infromation Statement Explanation'].replace(replace_map_result).replace(['', None, np.nan], 'N.A.')
        df_ai_pandas_filtered['iAuidt Score: Infromation Statement Explanation']= np.where(
            df_ai_pandas_filtered['iAuidt Result: Infromation Statement Explanation'].str.contains('N.A.'),10, df_ai_pandas_filtered['iAuidt Score: Infromation Statement Explanation']
        )

        print("Remark Count: ", df_ai_pandas_filtered['iAudit Remarks: Infromation Statement Explanation'].value_counts())
        export_to_excel(
            df_ai_pandas_filtered,
            f"voice_information_statement_{formatted_date if 'formatted_date' in locals() else datetime.now().strftime('%Y-%m-%d')}",
            "voice",
            "InformationStatementExplanation"
        )

        # final_res = spark.createDataFrame(df_ai_pandas_filtered)
        # final_res = final_res.toDF(*[c.replace(" ", "_") for c in final_res.columns])
        # final_res = final_res.withColumn("call_date", F.to_timestamp(F.lit(formatted_date), "yyyy-MM-dd"))

        # try:
        #     filtered_table = catalog_config['intermediate']['information_provided']
        #     date_exists = (
        #         spark.table(filtered_table)
        #         .filter(F.col("call_date") == formatted_date)
        #         .limit(1)
        #         .count() > 0
        #     )
        #     if date_exists:
        #         log.info(f"Results for {formatted_date} already exists. Deleting old data before insert.")
        #         spark.sql(f"DELETE FROM {filtered_table} WHERE call_date = '{formatted_date}'")
        # except Exception as e:
        #     log.error(f"Error in deleting old data: {e}, continuing with the insert")       

        # try:
        #     final_res.write.format("delta").option("mergeSchema","true").mode("append").saveAsTable(catalog_config['intermediate']['information_provided'])
        # except Exception as e:
        #     log.error(f"Error in writing to table: {e}\nSchema of final_res: {final_res.printSchema() if 'final_res' in locals() else 'final_res not defined'}")
        #     raise e

            
    except Exception as e:
        log.error(f"Module name: Information Statement Explanation- Error: {e}", exc_info=True)
        raise
            
   