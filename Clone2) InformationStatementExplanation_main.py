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
sys.path.append('../')
from utils.iaudit_logger import get_logger
from prompts.InformationStatementExplanationPrompts import *
from pyspark.sql.functions import format_string
from pyspark.sql.functions import array, lit, when, size, concat_ws
from pyspark.sql import functions as F
import yaml


warnings.filterwarnings("ignore")

# transcript_path = r'/Workspace/Users/ananthagiri_abhiram@next.co.uk/iAudit_Trisha/AllAuditedTransciptNonRedacted2026.xlsx'

# #one day call
# transcript_path = r'/Workspace/Users/ananthagiri_abhiram@next.co.uk/iAudit_Trisha/combined_file_2026_03_27.xlsx'

# result_path = r"InformationExplanationExplanationFinalResults27032026V2.xlsx"
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
        # formatted_date = '2026-03-19'
        try:
            # formatted_date = dbutils.jobs.taskValues.get(taskKey="Pre-Modules-Run", key="sql_date")
            formatted_date = '2026-05-25'
        except Exception as e:
            formatted_date = sys.argv[1]
            if not re.match(r"^\d{4}-\d{2}-\d{2}$", str(formatted_date)):
                raise ValueError(f"formatted_date '{formatted_date}' is not in yyyy-mm-dd format")
            logger.info(f"formatted_date: {formatted_date}")
        INPUT_FILE_PATH = spark.table(catalog_config['output_table']['filtered_table']).filter(f"call_date = '{formatted_date}'")
        transcripts =  INPUT_FILE_PATH.toPandas()

        # callids = ["08894915-9610-44d0-b8dd-e2206eef8c6a","0a2a738f-0a3d-4408-8cb9-9cee0b03d6d2","0cbee79b-87aa-447b-a7c8-6340de192fb7","0ce4c45c-b838-4237-b5eb-b1749a9b0c11","0cf04898-520f-4a87-938c-c8cfd054803c","0d479812-2dbd-44a1-bbf8-9c6d63170577","0d7b142f-2e44-4785-9243-720f741f15af","0ee7c7ed-e264-4fc8-a81e-f6cc855a8a51","123ec80c-5cef-42b4-ad0c-0a902a2f1435","155f0831-650f-4117-ba38-2895336fadc9","1e9e4579-e177-4c86-a2b9-79d8ce849bbb","1f4f5c56-74f8-471f-ae35-11cf5824f5a1","22f13ec4-d3e0-457c-b7e9-df3454e934d6","23e6d628-d9e2-4efb-a919-2804b8fb2324","286af837-0f07-418d-8deb-cb829ce65dec","29807d70-a595-42bd-9cf8-f65588cb7dc3","29b9eeaf-9bb0-4eb1-baaf-fba7d25aefdd","2a863a4c-6581-4802-ad98-db00773ac2a7","2cec97a3-ce9f-4440-9aa7-7518eebeffa0","2e65d592-a930-42d2-9b29-257f807efed0","321bd10d-8a56-434e-8124-97d8904faed5","36ce4997-54b2-438d-8e68-2267ec6c4351","40ecf721-b668-4f64-ace4-5896499c727c","41f845d5-df2b-4528-9ce4-c06b5c79871d","425bd272-f6f8-4da2-bfb2-8d2d77ce5054","4be261a7-6521-483c-8d5a-2f620fa744e9","4d9f4316-c877-4cd3-8cc7-177bead94153","4f59a4ba-601f-46f6-86b4-c86c18a9fa39","501d0e13-6cd1-44a6-821c-5db24a1068af","505129a0-4652-45b0-932f-20c4293ff958","5284dece-7ab0-4850-b5c6-648cfdc4e6af","52b4710f-7e1c-4d29-a677-32b86a5a8a80","543280a1-50c1-4bc2-b887-0a77d166914b","5595979c-fb17-48c2-9d4e-127f2e6fd702","57e1d411-a931-4071-abdd-3eca3300c26b","5d1c9000-315f-4906-a133-586946d1481e","62614f64-ce6b-4b48-8fbd-0c185451744d","64037133-2dfd-4e74-a80e-d7c5f5b02a32","6573802e-7398-4d4b-9dc9-bd51259415b1","67068558-249b-4a2a-ab9f-2af6bbe5203f","6cc8c760-4bdb-4123-9d81-32ff98239a93","6d6558ec-244a-4f11-a391-e1551c19f940","7125fd30-cb94-4e87-af09-afa358662cd3","76131899-b147-46a5-82d5-a932f7330ab0","7671eb19-994f-45d3-b2e1-3e5578da8772","7e48141f-cd8c-4a1f-85d5-9ab168f4266c","802c2fe5-d865-4725-ae0e-6190eaa427c5","83b4e94f-6e06-4aab-a314-2c82dd3942f4","853318f3-06ec-42b8-8563-598a7959b675","8718e2d8-27e9-4e1b-b112-c600ec83fc0f","8873658d-d980-46f6-af47-f1f8b447810f","8911754b-52c3-4d15-b84b-8d63b2fb94fc","8c042ccc-106a-420c-a030-12b390a84a70","8dd4f722-2483-4d57-b51f-54b70e61c296","934dc6a5-7e32-42dc-ae3f-72a8aa795213","955f551d-7d48-472b-8601-c35b5c0c4ef0","99d606bc-e783-4716-98bf-1bf7e411b6cc","9a2b048e-4f46-4059-941c-8b366e24e8d2","9cc16fdb-2040-4c7e-912d-400e3a1ebe6b","a19948e3-72ae-4aa3-9859-7930b6b1d147","a45c56d2-be88-40a4-bccf-8c849e196910","a766f06a-a7dc-4f82-a1b0-a4acee6e121c","a7c9f9e2-4d5d-40ac-bbff-f42451149961","aaabe31b-d7b1-48cd-92ec-32eeb16bf0f0","ae98aa76-8f72-4f70-92fc-75b3c7b96a7f","aef5d770-e686-498f-9602-0d3c4e33f3de","b29662b3-cd4f-479f-8d55-ba7924217354","b7a5eff7-95c3-48db-9579-76bb42016c9e","b89d4dc8-4b28-409f-bfda-9cd958a1a366","c049a487-c9a7-4745-b4ca-481b2b146bc3","c2c07abd-f771-4a4d-b85d-fd080abab9ad","c44a77dd-4b2b-4df8-a0c9-b57cc2a2c5fb","c8c7adca-a99f-4938-9fdd-2252a35a2239","d092d320-dcc4-4668-be37-e85f3fc69380","d79a114a-62d8-4d70-8988-454171bbee7a","dd3f2655-465e-49f7-8cb9-2a327fd8bdb5","dfa7a95d-225c-40b7-9657-6cde4d3315e3","e024cdee-a48c-437c-bb07-2fa0fe04e2f8","e86b9288-21a3-49c6-962a-ed84b01a8e22","eec80d2e-416c-4cdb-a7de-a056e2095d2a","f42bc7df-4325-4602-adb4-a93984e1f1b9","f65748de-d8fd-43bf-b513-bd8fbc615383","f79d1e74-8ff5-49a7-befc-c4180ebb1a00","f8016b6b-f710-44ef-b4df-f4ed25a1df92","fb4d033f-fc1a-4e53-8f2d-10010f9af13b","fcff56c9-b3a6-4cc9-b197-59919a29aff2","feaea2c4-d577-439d-b95c-07c435308485","fec6a2d4-f26b-4f59-bbbd-05d1dbee820c"]

        # transcripts = transcripts[transcripts['callid'].isin(callids)].reset_index(drop=True)  

        transcripts =transcripts.rename(columns={'AgentID':'AgentId'})
  

        total_calls = len(transcripts['callid'].unique().tolist())
        filtered_calls = transcripts['callid'].unique().tolist()[:total_calls]
        # filtered_calls = transcripts['callid'].unique().tolist()[:100]
        df_transcripts = transcripts[transcripts['callid'].isin(filtered_calls)].reset_index(drop=True)
    
        audit_callids = df_transcripts['callid'].unique().tolist()

        # df_transcripts['createdate'] = pd.to_datetime(df_transcripts['createdate'], format = 'mixed', errors='coerce')
        # df_transcripts = df_transcripts.sort_values(by=["callid", 'createdate'])
        # df_transcripts['createdate'] = df_transcripts['createdate'].astype(str)

        spark_logger =  spark.sql(f"""
                SELECT `WorkGroupName`, `conversation_id`, `ResponseName`,account_number FROM contactcentre_prod.staging.zen_live9 
                WHERE conversation_id IN {tuple(audit_callids)}      
            """)

        df_logger = spark_logger.toPandas()

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
            # &(df_ai_pandas["Did the transcript mention refund of charges?"].str.lower()=='no')
                      )
            , 'Compliant with development' , 'No Error')
        
        # Addditional check for payment calls
        df_ai_spark = spark.createDataFrame(df_ai_pandas)
        prompt_email_confirmation_json = json.dumps(prompt_payment_email_confirmation)
        
        df_ai_spark = (
            df_ai_spark.withColumn(
                "ai_response_payment_email", F.when(
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
            (df_ai_pandas['payment_email_check_indicator'] == 1) &
            (df_ai_pandas['Result: Successful Make Payment Email'] == 'No Error')
        )
        df_ai_pandas.loc[mask1, 'ai_response_payment_email'] = (
            df_ai_pandas.loc[mask1, 'ai_response_payment_email'].apply(safe_repair_and_load)
        )

        df_ai_pandas = pd.concat(
            [
                df_ai_pandas,
                df_ai_pandas['ai_response_payment_email'].apply(pd.Series).rename(columns={'Did advisor mention that a confirmation of payment will be sent to registered email id/address?':'Did advisor mention that a confirmation of payment will be sent to registered email id/address_confirmation?'})
            ],
            axis=1
        )


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
        df_ai_pandas['iAuidt Score: Infromation Statement Explanation']= np.where(
            df_ai_pandas_filtered['iAuidt Result: Infromation Statement Explanation'].str.contains('N.A.'),10, df_ai_pandas_filtered['iAuidt Score: Infromation Statement Explanation']
            )

        df_ai_pandas_filtered.to_excel("../temp/information_statement_op.xlsx", index=False)
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
            
   