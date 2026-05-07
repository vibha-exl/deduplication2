# import os,sys
# sys.path.append('/Workspace/Users/pawan_kumar@next.co.uk/iAudit_deployment')

# !pip install  json-repair  -q
# !pip install rapidfuzz
import warnings
from rapidfuzz import fuzz, process
from json_repair import repair_json
import os
import json
import re
import numpy as np
from datetime import datetime
import pandas as pd
from pyspark.errors import PySparkException
from functools import reduce
from pyspark.sql import DataFrame
from pyspark.sql.types import StructType, StructField, BooleanType, StringType
from pyspark.sql.functions import from_json, col, lit, udf, when, expr
from iaudit_logger import get_logger
from AccountAmendmentClosurePrompts import *
from pyspark.sql.functions import format_string
from pyspark.sql.functions import array, lit, when, size, concat_ws
from pyspark.sql import functions as F
import yaml
warnings.filterwarnings("ignore")
from openpyxl.utils import escape

def remove_illegal_chars(val):
    if isinstance(val, str):
        # Remove illegal characters for Excel (control chars except \t, \n, \r)
        return re.sub(r"[\x00-\x08\x0b\x0c\x0e-\x1f]", "", val)
    return val

# transcript_path = r'/Workspace/Users/ananthagiri_abhiram@next.co.uk/iAudit_Trisha/NonRedactedUAT.xlsx'

# transcript_path = r'/Workspace/Users/ananthagiri_abhiram@next.co.uk/iAudit_Trisha/AccountAmendmentMarkdownCalls27Nov.xlsx'
# transcript_path = r'/Workspace/Users/ananthagiri_abhiram@next.co.uk/iAudit_Trisha/all04122accountamendment.xlsx'
# transcript_path = r'/Workspace/Users/ananthagiri_abhiram@next.co.uk/iAudit_Trisha/AuditedAccountAmendmentTranscripts.xlsx'

#pre audited call all
# transcript_path = r'/Workspace/Users/ananthagiri_abhiram@next.co.uk/iAudit_Trisha/AllAuditedTransciptNonRedacted2026.xlsx'

#one day call
# transcript_path = r'/Workspace/Users/ananthagiri_abhiram@next.co.uk/iAudit_Trisha/combined_file_2026_03_03.xlsx'

# transcript_path = r'/Workspace/Users/ananthagiri_abhiram@next.co.uk/iAudit_Trisha/combined_file_2026_03_20 (1).xlsx'

#na_close_accounts
transcript_path = r'/Workspace/Users/ananthagiri_abhiram@next.co.uk/iAudit_Trisha/NACloseAccount356availT.xlsx'


# /Workspace/Users/ananthagiri_abhiram@next.co.uk/iAudit_Trisha/0512AllcallsforAccountAmendment.xlsx
result_path = r"AccountAmendmentClosureFinalResutsNACloseAccountsLargeLLM.xlsx"

log = get_logger()
# log.error(f"module name: Account Amendment and Closure: {e}", exc_info=True)





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
        transcripts = pd.read_excel(transcript_path)
        #transcripts = pd.read_csv(transcript_path)

        # formatted_date = '2025-12-07'
        # raw_table_query = f"""SELECT * FROM contactcentre_prod.transcripts.transcripts_raw WHERE DATE(createdate) = '{formatted_date}'"""
    
        # full_raw_table = spark.sql(raw_table_query)
        # transcripts = full_raw_table.toPandas()
        # transcripts =transcripts.rename(columns={'AgentID':'AgentId'})
        

        total_calls = len(transcripts['callid'].unique().tolist())
        filtered_calls = transcripts['callid'].unique().tolist()[:total_calls]
        # filtered_calls = transcripts['callid'].unique().tolist()[:100]
        df_transcripts = transcripts[transcripts['callid'].isin(filtered_calls)].reset_index(drop=True)
        audit_callids = df_transcripts['callid'].unique().tolist()

        # df_transcripts['createdate'] = pd.to_datetime(df_transcripts['createdate'], format = 'mixed', errors='coerce')
        # df_transcripts = df_transcripts.sort_values(by=["callid", 'createdate'])
        # df_transcripts['createdate'] = df_transcripts['createdate'].astype(str)

        num_unique_callid_agentid = transcripts[['callid', 'AgentId']].drop_duplicates().shape[0]
        print("num_unique_callid_agentid transcript", num_unique_callid_agentid)

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
        print("Logger unique callids", len(df_logger['CallId'].unique().tolist()))
        customer_account_number_list = df_logger['account_number'].unique().tolist()
        customer_account_number_list = [x for x in customer_account_number_list if str(x) != 'nan']
        print("unique_account_number",len(customer_account_number_list))
        customer_account_number_list = [ x for x in customer_account_number_list if x is not None]
        print("Not none unique_account_number",len(customer_account_number_list))

        spark_account_details =  spark.sql(f"""
        SELECT `account_number`, `AccountType`, `SpecialAccountDescription`
        FROM `contactcentre_prod`.`iaudit`.`account_details`
        WHERE `account_number` IN {tuple(customer_account_number_list)}  
        """)
        df_account_details = spark_account_details.toPandas()
        print(df_account_details.shape)
        df_account_details= df_account_details.drop_duplicates(subset = 'account_number', keep='first')
        print("Account details final",df_account_details.shape)

        df_account_details.to_excel("account_details_for_account_amnedemnt_check.xlsx")
        account_number_account_details_list = df_account_details['account_number'].unique().tolist()
        print("Account Details unique account number", len(account_number_account_details_list))
        # df_account_details = df_account_details.rename(columns={'AccountNumber': 'account_number'})
        df_logger = df_logger.merge(df_account_details, on='account_number', how='left')
        df_logger['account_number'] = df_logger['account_number'].astype(str)
        df_logger['AccountType'].replace('',np.nan, inplace = True)
        df_logger['AccountType'].fillna('N.A.', inplace = True)
        df_logger['SpecialAccountDescription'].replace('',np.nan, inplace = True)
        df_logger['SpecialAccountDescription'].fillna('N.A.', inplace = True)
        print("Logger shape after join with account details",df_logger.shape )


    

        spark_account_hold_dd =  spark.sql(f"""
            SELECT `AccountNumber`, `HoldOrderIndicator`, `DirectDebit`
            FROM `contactcentre_prod`.`iaudit`.`account_hold_dd`
            WHERE `AccountNumber` IN {tuple(customer_account_number_list)}  
        """)
        df_account_hold_dd = spark_account_hold_dd.toPandas()
        df_account_hold_dd = df_account_hold_dd.rename(columns={'AccountNumber': 'account_number'})
        df_logger = df_logger.merge(df_account_hold_dd, on='account_number', how='left')
        df_logger['account_number'] = df_logger['account_number'].astype(str)
        # print(df_logger.columns)
        # print("hold data",df_account_hold_dd.shape)
        # print("hold data unique account_number",len(df_account_hold_dd['account_number'].unique().tolist()))

        spark_vw_customers =  spark.sql(f"""
            SELECT `AccountNumber`,  `EffectiveEndDate`, `Gender`,
            `Title`,`Initial`,`Surname`,`Forename`,`ForenameClean`,
        `AddressLine1`, `AddressLine2`,`AddressLine3`,`AddressLine4`,`AddressLine5`,`AddressLine6`, `HomePhone`,`MobilePhone`,`ContactPhone`,`EmailAddress`, `DateOfBirth`
            FROM  `businessintelligencesystems_prod`.`online_pii_transformed`.`vw_customers`
            WHERE `AccountNumber` IN {tuple(customer_account_number_list)}  
        """)
        df_vw_customers= spark_vw_customers.toPandas()
        df_vw_customers = df_vw_customers.rename(columns={'AccountNumber': 'account_number'})
        print("df_vw_customers", df_vw_customers.shape)
        df_vw_customers['EffectiveEndDate']= pd.to_datetime( df_vw_customers['EffectiveEndDate'], errors ='coerce')
        df_vw_customers= df_vw_customers.groupby('account_number', group_keys=False).apply(select_row_effect_end_date_condition).reset_index( drop=True)
        # print('df_vw_customers after',df_vw_customers.shape)
        

        df_logger = df_logger.merge(df_vw_customers, on='account_number', how='left')
        df_logger['account_number'] = df_logger['account_number'].astype(str)
        # print(df_logger.columns)
        # print("customer data",df_vw_customers.shape)
        # print("customer data unique account_number",len(df_vw_customers['account_number'].unique().tolist()))

        level_2_req_list = ['Change Of Detail - Address','Change Of Address Used Without Consent', 'Change Of Detail - Telephone Number','Change Of Detail - Email Address','Change Of Detail - Name Or Title', 'Close Account - No Reason' ,'Close Account - Service','Gone Away Set' ] #'Gone Away Set'
        df_logger['priority'] =df_logger['Level 2'].isin(level_2_req_list).astype(int)
        # print("Logger before",df_logger.shape)
        df_logger_unique = (df_logger.sort_values(['CallId', 'priority'], ascending=[True, False]).drop_duplicates(subset = 'CallId', keep='first').drop(columns='priority'))
        # print(df_logger_unique.shape)

        data = pd.merge(df_transcripts, df_logger_unique, left_on='callid', right_on='CallId', how='left')
        data['callid_agentid']= data['callid'] + "|" + data['AgentId']
        # print("data",data.columns)
        grouped_transcript_all = data.groupby('callid_agentid').apply(lambda x: "\n".join(f"{row['channel']}: {row['transcript']}" for _, row in x.iterrows())).reset_index(name = 'transcript')
        data_intermediate = data.groupby('callid_agentid', as_index = False).first()
        data_req = pd.merge(grouped_transcript_all, data_intermediate, on='callid_agentid', how='left')
        data_req['createdate'] = pd.to_datetime(data_req['createdate'], format = 'mixed', errors='coerce')
        print("create_date",data_req['createdate'].value_counts() )
        data_req = data_req.sort_values(by=["callid", 'createdate'], ascending =[True, True])
        data_req['createdate'] = data_req['createdate'].astype(str)
        data_req['agent_order'] = data_req.groupby("callid").cumcount()+1
        data_req['agent_order_max'] = data_req.groupby("callid")["agent_order"].transform("max")
        data_req['customer_name_transformed'] = data_req['Title'] + " " +data_req['Forename'] + " " + data_req['Surname']
        data_req['customer_address_transformed'] =  data_req['AddressLine2'] + " " +data_req['AddressLine3'] + " " +data_req['AddressLine4'] + " " +data_req['AddressLine5'] + " " +data_req['AddressLine6']
        data_req['customer_phone_transformed'] = "Home Phone: "+data_req['HomePhone']+ ", Mobile Phone: " + data_req['MobilePhone']+ ", Contact Phone: "+data_req['ContactPhone']
        
        #print("final data",data_req.shape)
        log.info(f"Final data shape {data_req.shape}")


    
        level_2_req_list = ['Change Of Detail - Address','Change Of Address Used Without Consent', 'Change Of Detail - Telephone Number','Change Of Detail - Email Address','Change Of Detail - Name Or Title', 'Close Account - No Reason' ,'Close Account - Service']
    
        # data_req = data_req[~data_req['account_number'].isin(['','UNKNOWN']) & data_req['account_number'].notna() ]
        # data_req = data_req[data_req['account_type'].notna() & (data_req['account_type']!='')]

        cols_to_check_null = ['account_number', 'account_type']
        data_req['account_number'].replace('',np.nan, inplace = True)
        data_req['account_number'].fillna('N.A.', inplace = True)
        data_req['account_type'].replace('',np.nan, inplace = True)
        data_req['account_type'].fillna('N.A.', inplace = True)


        print("after removing null",data_req.shape )



        #  maintpulating to get llm hits for stored data (remove in final code)
        # data_req['Level 2'] = level_2_req_list * (len(data_req) // len(level_2_req_list)) + level_2_req_list[:(len(data_req) % len(level_2_req_list))]
        #print("value counts",data_req.head(10)['Level 2'].value_counts())

        level_2_to_validation_columns = {
                                'Change Of Detail - Address':'customer_address_transformed',
                        'Change Of Address Used Without Consent' :'customer_address_transformed',
                            'Change Of Detail - Telephone Number':'customer_phone_transformed',
                                        'Change Of Detail - Email Address':'EmailAddress',
                                    'Change Of Detail - Name Or Title':'customer_name_transformed'}
        conditions = [data_req['Level 2']==k for k in level_2_to_validation_columns.keys()]
        choices = [data_req[v] for v in level_2_to_validation_columns.values()]

        data_req['validation_data'] = np.select(conditions, choices, default="") 

        # print("val data",data_req['validation_data'].value_counts())


        df_prompts = (
                        spark.createDataFrame(data_req).withColumn(
                            "user_prompt", F.when((F.col("Level 2")=="Change Of Detail - Name Or Title") & (F.col("account_type")=="credit"), F.lit(user_prompt_name_change))
                            .when((F.col("Level 2")=="Change Of Detail - Name Or Title") & (F.col("account_type")=="cash"), F.lit(user_prompt_name_change_cash_account))
                            .when(F.col("Level 2")=="Change Of Detail - Address", F.lit(user_prompt_address_change))
                            .when(F.col("Level 2")=="Change Of Address Used Without Consent", F.lit(user_prompt_address_change))
                            .when(F.col("Level 2")=="Change Of Detail - Telephone Number", F.lit(user_prompt_telephone_change))
                            .when(F.col("Level 2")=="Change Of Detail - Email Address", F.lit(user_prompt_email_change))
                            .when(F.col("Level 2")=="Close Account - No Reason", F.lit(user_prompt_account_closure))
                            .when(F.col("Level 2")=="Close Account - Service", F.lit(user_prompt_account_closure))
                            .otherwise(F.lit(""))

                        )

                        .withColumn("final_prompt", 
                                    F.when((F.col("Level 2") == "Close Account - No Reason") |(F.col("Level 2") == "Close Account - No Reason") ,
                                            F.concat_ws( "\n\n",
                                                        F.lit("SYSTEM PROMPT:"),
                                                        F.lit(system_prompt),
                                                        F.lit("USER PROMPT:"),
                                                        F.col("user_prompt"),
                                                        F.lit("TRANSCRIPT:"),
                                                        F.col("transcript_x")
                                                        )
                                            ) .otherwise(
                                                F.concat_ws( "\n\n",
                                                        F.lit("SYSTEM PROMPT:"),
                                                        F.lit(system_prompt),
                                                        F.lit("USER PROMPT:"),
                                                        F.col("user_prompt"),
                                                        F.lit("TRANSCRIPT:"),
                                                        F.col("transcript_x"),
                                                        F.lit("VALIDATION DATA:(Database)"),
                                                        F.col("validation_data")
                                                        
                                                        )
                                            )
                                            )
                        

        )
        # databricks-meta-llama-3-1-8b-instruct databricks-meta-llama-3-3-70b-instruct #contact_centre_internal_batch
        df_ai = (
            df_prompts.withColumn(
                "ai_response", F.when(
                    (F.col("Level 2").isin(level_2_req_list)) &
                (F.col("agent_order_max")== F.col("agent_order")) &
                (F.col("user_prompt") != ""), F.expr(
                    """ai_query('databricks-meta-llama-3-3-70b-instruct',request => final_prompt)"""
                        )
                
            )
        )
        )
        #display(df_ai.toPandas())
    # df_ai.toPandas().to_excel("AccountAmendemntIntermediateResults.xlsx")

        df_ai_pandas = df_ai.toPandas()
        print("LLM hits done",df_ai_pandas.shape)
        if 'ai_response' not in df_ai_pandas.columns:
                df_ai_pandas['ai_response'] = ""

        df_ai_pandas['ai_response1'] = (df_ai_pandas['ai_response']
                                        .str.replace("```json", "", regex = False)
                                        .str.replace("```", "",regex = False)
                                        .str.strip()
                                        )
        df_ai_pandas['Output_final'] = df_ai_pandas['ai_response1'].apply(safe_repair_and_load)
        df_ai_pandas = pd.concat([df_ai_pandas,df_ai_pandas['Output_final'].apply(pd.Series)], axis =1) 
        #df_ai_pandas.to_excel("AccountAmendemntBeforeFinalScoring.xlsx")

        #Final Scoring
        log.info(f"Final scoring  happening:")

        ai_response_column_list = [
            "Does the conversation involve name change?",
            "Was the name change validation correct?",
            "Evidence",
            "Does name change involve only title changes (like miss Mrs Mr)?",
            "If full name change/first name change, did agent ask to close account?",
            "Does the conversation involve telephone change?",
            "Was the telephone change validation correct?",
            "Does the conversation involve no change due to 24hrs security?",
            "Does the conversation involve address change?",
            "Did the agent ask if caller wants to place an order today?",
            "Did the agent ask caller to allow until tomorrow before placing next order?",
            "Did the agent ask if caller has access to the mobile/phone number?",
            "Did the agent ask any additional security questions or digits of card?",
            "Did the pin verification through phone fail?",
            "If the pin verification fail, were the security questions asked?",
            "Did the agent ask the caller if contact number needs to be changed?",
            "Did the agent ask the caller if email address needs to be changed?",
            "Did the agent confirm address change and that a letter will be sent to new address?",
            "Was the address change validation correct?",
            "In the conversation is the address change proccess actually followed/change done during call?",
            "Was any PIN discussed in the conversation?",
            "Did the conversation involve discussion over caller's registered telephone number/mobile number/any phone?",
            "Was there any conversation regarding placing orders/placed orders?",
            "Was there any discussion about order being placed/re-ordered/cancelled/processed?",
            "Was there any discussion about any orders?",
            "Does the caller wants to place an order?",

            "Does the conversation involve email change?",
            "Does conversation about any previous phone change so additional security for email?",
            "Did the agent ask about the phone being accessible for pin verification?",
            "Did the pin verification through phone happen for email change?",
            "If the pin verification fail, did the agent ask security questions?",
            "After pin verification, did agent confirm email change?",
            "Was the email change validation correct?",
            "Does part of conversation have account closure?",
            "Did the agent ask about pending orders?",
            "Did the caller have any pending orders?",
            "If the order is pending, did the agent ask the caller to call back after order delivery?",
            "Did the agent ask about any refund or items to return?",
            "Did the caller have pending returns/refunds?",
            "If the returns/refund is pending, did the agent ask the caller to call back after?",
            "Did the agent ask for the reason for closure?",
            "Did agent ask for the account closure confirmation?",
            "Was the account closure done?",
            "Did the name change happen during conversation/trancript?",
            "Did the name in transcript even slightly match the database?",
            "Did the conversation have anything about pin verification failing/did pin verification fail?",
            "Did the email change happen during the conversation/transcipt?",
            "Did the phone/cell/contact/mobile number change happen during conversation/trancript?",
            "Did the phone/cell/contact/mobile number in transcript even slightly match the database?",
            "Is the conversation in the transcript partial/ended abruptly/incomplete conversation?",
            "Did the agent mention that the pin verification fail?",
            "Did the agent ask if they have resolved and said bye?"


        ]
        for col in ai_response_column_list:
                if col not in df_ai_pandas.columns:
                    df_ai_pandas[col] = "N.A."

        df_ai_pandas[ai_response_column_list] = df_ai_pandas[ai_response_column_list].replace("", np.nan).fillna("N.A.")          


        df_ai_pandas['iAudit Result: Account Amendment and Closure']= "N.A."
        df_ai_pandas['iAudit Score: Account Amendment and Closure']= 10	
        df_ai_pandas['iAudit Remarks: Account Amendment and Closure']=	""


        #df_ai_pandas.to_excel("AccountAmendemntAfterInitialisation.xlsx")



        ####Account Closure################

        df_ai_pandas['AccountClosure'] = np.where(

        ((df_ai_pandas['Level 2'] != 'Close Account - No Reason' )|(df_ai_pandas['Level 2'] != 'Close Account - Service' ) | (df_ai_pandas['Does part of conversation have account closure?'].str.lower()== 'no')  )
            
            ,"N.A." ,'')

            
        df_ai_pandas['Result: No pending order to be delivered_(AccountClosure)'] = np.where(

        (df_ai_pandas['Did the agent ask about pending orders?'].str.lower() == 'no')    
            
            ,'Amber Error' , 'No Error')
            
        df_ai_pandas['Result: No pending order to be delivered_(AccountClosure)'] = np.where(

        ((df_ai_pandas['AccountClosure']== 'N.A.' ) )  
            
            ,'N.A.' , df_ai_pandas['Result: No pending order to be delivered_(AccountClosure)'])


        df_ai_pandas['Result: Pending returns or refund_(AccountClosure)'] = np.where(

        (df_ai_pandas['Did the agent ask about any refund or items to return?'].str.lower() == 'no')   
            
            ,'Red Error' , 'No Error')

        df_ai_pandas['Result: Pending returns or refund_(AccountClosure)'] = np.where(

        ((df_ai_pandas['AccountClosure']== 'N.A.' ) )  
            
            ,'N.A.' , df_ai_pandas['Result: Pending returns or refund_(AccountClosure)'])




        df_ai_pandas['Result: Account closure reason_(AccountClosure)'] = np.where(

        (df_ai_pandas['Did the agent ask for the reason for closure?'].str.lower() == 'no')   
            
            ,'Amber Error' , 'No Error')

        df_ai_pandas['Result: Account closure reason_(AccountClosure)'] = np.where(

        ((df_ai_pandas['AccountClosure']== 'N.A.' ) )  
            
            ,'N.A.' , df_ai_pandas['Result: Account closure reason_(AccountClosure)'])





        df_ai_pandas['Result: Account closure communication_(AccountClosure)'] = np.where(

        (df_ai_pandas['Did agent ask for the account closure confirmation?'].str.lower() == 'no' )  
            
            ,'Amber Error' , 'No Error'
            
        )
        df_ai_pandas['Result: Account Closure DirectDebit']=np.where((df_ai_pandas['Was the account closure done?'].str.lower()== 'yes') & (df_ai_pandas['DirectDebit'] != 'N') & (df_ai_pandas['Does part of conversation have account closure?'].str.lower()== 'yes'),  'Red error', 'No Error')

        df_ai_pandas['Result: Account Closure DirectDebit']=np.where(

        ((df_ai_pandas['AccountClosure']== 'N.A.' ) )  
            
            ,'N.A.' , df_ai_pandas['Result: Account Closure DirectDebit'])

        df_ai_pandas['Result: Account closure communication_(AccountClosure)'] = np.where(

        ((df_ai_pandas['AccountClosure']== 'N.A.' ) )  
            
            ,'N.A.' , df_ai_pandas['Result: Account closure communication_(AccountClosure)'])

        # df_ai_pandas['AccountClosure'] = np.where(

        # ((df_ai_pandas['AccountClosure']== 'N.A.' )
        # |((df_ai_pandas['Result: Account closure communication_(AccountClosure)'] == 'No Error')

        # & (df_ai_pandas['Result: Account closure reason_(AccountClosure)'] == 'No Error')
        # & (df_ai_pandas['Result: Pending returns or refund_(AccountClosure)'] == 'No Error')
        # &  (df_ai_pandas['Result: No pending order to be delivered_(AccountClosure)'] == 'No Error')
        # &  (df_ai_pandas['Result: Account Closure DirectDebit'] == 'No Error'))
        

        
        # )
            
        #     ,'NO MARKDOWN' , 'MARKDOWN')


        result_cols = [col for col in df_ai_pandas.columns if col.startswith("Result:") ]        
        df_ai_pandas['Result: Account Closure'] =  df_ai_pandas[result_cols].apply(get_result_priority, axis = 1)
        df_ai_pandas['Score: Account Closure'] = df_ai_pandas['Result: Account Closure'].map(score_map)

        df_ai_pandas['Score: Account Closure'] = np.where(
            df_ai_pandas['Result: Account Closure'] == "N.A." , 10, df_ai_pandas['Score: Account Closure'])
        #df_ai_pandas.to_excel("AccountAmendemntAfterclosure.xlsx")
        ############### change of name #################################

        df_ai_pandas['Result: Account Amendment- Name Change Validation'] = np.where(

        (df_ai_pandas['Was the name change validation correct?'].str.lower() == 'no' ) & 
        (df_ai_pandas["Does the conversation involve name change?"].str.lower() == 'yes' ) &
        (df_ai_pandas["Did the name change happen during conversation/trancript?"].str.lower() == 'yes' ) & (df_ai_pandas['Did the name in transcript even slightly match the database?'].str.lower() == 'no' )
            
            ,'Red Error' , 'No Error')
        
        df_ai_pandas['Result: Account Amendment- Name Change Validation'] = np.where(
        (df_ai_pandas["Does the conversation involve name change?"].str.lower() == 'no') | (df_ai_pandas['Level 2'] != 'Change Of Detail - Name Or Title' )
        , 'N.A.',df_ai_pandas['Result: Account Amendment- Name Change Validation'] )
        


        df_ai_pandas['Result: Account Amendment- Name Change (Cash account, ask to close account)'] = np.where(

        ((df_ai_pandas['If full name change/first name change, did agent ask to close account?'].str.lower() == 'no' ) & (df_ai_pandas["Does the conversation involve name change?"].str.lower() == 'yes' )&(df_ai_pandas["Did the name change happen during conversation/trancript?"].str.lower() == 'yes' ))
            
            ,'Red Error' , 'No Error')


        df_ai_pandas['Result: Account Amendment- Name Change (Cash account, ask to close account)'] = np.where(

        (df_ai_pandas['account_type'].str.lower() == 'cash' ) &((df_ai_pandas["Does the conversation involve name change?"].str.lower() == 'no') | (df_ai_pandas['Level 2'] != 'Change Of Detail - Name Or Title' )) 
            ,df_ai_pandas['Result: Account Amendment- Name Change (Cash account, ask to close account)'] , 'N.A.')
            
            
        result_cols = [col for col in df_ai_pandas.columns if col.startswith("Result:") ]            
        df_ai_pandas['Result: Account Amendment- Name Amendment'] =  df_ai_pandas[result_cols].apply(get_result_priority, axis = 1)

        df_ai_pandas['Result: Account Amendment- Name Amendment'] = np.where(
        (df_ai_pandas["Does the conversation involve name change?"].str.lower() == 'no') | (df_ai_pandas['Level 2'] != 'Change Of Detail - Name Or Title' )
        , 'N.A.',df_ai_pandas['Result: Account Amendment- Name Amendment'] )

        df_ai_pandas['Score: Account Amendment- Name Amendment'] = df_ai_pandas['Result: Account Amendment- Name Amendment'].map(score_map)

        df_ai_pandas['Score: Account Amendment- Name Amendment'] = np.where(
            df_ai_pandas['Result: Account Amendment- Name Amendment'] == "N.A." , 10, df_ai_pandas['Score: Account Amendment- Name Amendment'])    

        ############### change of phone #################################
        #

        df_ai_pandas['Result: Account Amendment- Phone Change Validation'] = np.where(

        ((df_ai_pandas['Was the telephone change validation correct?'].str.lower() == 'no' ) & (df_ai_pandas['Does the conversation involve telephone change?'].str.lower() == 'yes' )
         &(df_ai_pandas['Did the phone/cell/contact/mobile number change happen during conversation/trancript?'].str.lower() == 'yes' ) & (df_ai_pandas['Did the phone/cell/contact/mobile number in transcript even slightly match the database?'].str.lower() == 'no')
         )
            
            ,'Amber Error' , 'No Error')

        df_ai_pandas['Result: Account Amendment- Phone Change Validation'] = np.where(

        ((df_ai_pandas['Does the conversation involve telephone change?'].str.lower() == 'no' ) | (df_ai_pandas['Level 2'] != 'Change Of Detail - Telephone Number'))
            
            ,'N.A.' , df_ai_pandas['Result: Account Amendment- Phone Change Validation'])

        df_ai_pandas['Result: Account Amendment- Phone Change Validation'] = np.where(

        ((df_ai_pandas['Level 2'] == 'Change Of Detail - Telephone Number' )  )
            
            ,df_ai_pandas['Result: Account Amendment- Phone Change Validation'] , 'N.A.')

        result_cols = [col for col in df_ai_pandas.columns if col.startswith("Result:") ]        
        df_ai_pandas['Result: Account Amendment- Phone Amendment'] =  df_ai_pandas[result_cols].apply(get_result_priority, axis = 1)
        df_ai_pandas['Result: Account Amendment- Phone Amendment'] = np.where(

        df_ai_pandas['Level 2'] =='Change Of Detail - Telephone Number'
            
            ,df_ai_pandas['Result: Account Amendment- Phone Amendment'] , 'N.A.')

        df_ai_pandas['Score: Account Amendment- Phone Amendment'] = df_ai_pandas['Result: Account Amendment- Phone Amendment'].map(score_map)
        df_ai_pandas['Score: Account Amendment- Phone Amendment'] = np.where(
            df_ai_pandas['Result: Account Amendment- Phone Amendment'] == "N.A." , 10, df_ai_pandas['Score: Account Amendment- Phone Amendment'])

        ######################### change of address #####################################
        # "In the conversation is the address change proccess actually followed/change done during call?",
        #     "Was any PIN discussed in the conversation?",
        #     "Did the conversation involve discussion over caller's registered telephone number/mobile number/any phone?",
        #     "Was there any conversation regarding placing orders/placed orders?",
        #     "Was there any discussion about order being placed/re-ordered/cancelled/processed?",
        #     "Was there any discussion about any orders?",
        #     "Does the caller wants to place an order?",








        df_ai_pandas['Result: Account Amendment- Address Change ask place order'] = np.where(

        ((df_ai_pandas['Did the agent ask if caller wants to place an order today?'].str.lower() == 'no' ) & (df_ai_pandas['Does the conversation involve address change?'].str.lower() == 'yes' )
        & (df_ai_pandas['Was there any conversation regarding placing orders/placed orders?'].str.lower() =='no') & (df_ai_pandas['Was there any discussion about any orders?'].str.lower() =='no')
        & (df_ai_pandas['Does the caller wants to place an order?'].str.lower() =='no') & (df_ai_pandas['In the conversation is the address change proccess actually followed/change done during call?'].str.lower() =='yes')
         
         )
            
            ,'Amber Error' , 'No Error')


        df_ai_pandas['Result: Account Amendment- Address Change ask place order'] = np.where(

        ((df_ai_pandas['Does the conversation involve address change?'].str.lower() == 'no' )|((df_ai_pandas['Level 2'] != 'Change Of Detail - Address' )&(df_ai_pandas['Level 2'] != 'Change Of Address Used Without Consent' )))
            
            ,'N.A.' , df_ai_pandas['Result: Account Amendment- Address Change ask place order'])
        
        df_ai_pandas['Result: Account Amendment- Address Change ask place order'] = np.where((df_ai_pandas['Result: Account Amendment- Address Change ask place order']=='Amber Error')&(df_ai_pandas['transcript_x'].str.contains('place an order', case=False, na=False)),'No Error', df_ai_pandas['Result: Account Amendment- Address Change ask place order'] )

        df_ai_pandas['Result: Account Amendment- Address Change ask place order'] = np.where((df_ai_pandas['Result: Account Amendment- Address Change ask place order']=='Amber Error')&(df_ai_pandas['transcript_x'].str.contains('place any new order', case=False, na=False)),'No Error', df_ai_pandas['Result: Account Amendment- Address Change ask place order'] )

        df_ai_pandas['Result: Account Amendment- Address Change ask place order'] = np.where((df_ai_pandas['Result: Account Amendment- Address Change ask place order']=='Amber Error')&(df_ai_pandas['transcript_x'].str.contains('place new order', case=False, na=False)),'No Error', df_ai_pandas['Result: Account Amendment- Address Change ask place order'] )
        

        df_ai_pandas['Result: Account Amendment- Address Change ask place order'] = np.where((df_ai_pandas['Result: Account Amendment- Address Change ask place order']=='Amber Error')&(df_ai_pandas['transcript_x'].str.contains('placed the order', case=False, na=False)),'No Error', df_ai_pandas['Result: Account Amendment- Address Change ask place order'] )

        

        df_ai_pandas['Result: Account Amendment- Address Change ask place order'] = np.where((df_ai_pandas['Result: Account Amendment- Address Change ask place order']=='Amber Error')&(df_ai_pandas['transcript_x'].str.contains('made the order', case=False, na=False)),'No Error', df_ai_pandas['Result: Account Amendment- Address Change ask place order'] )

        df_ai_pandas['Result: Account Amendment- Address Change ask place order'] = np.where((df_ai_pandas['Result: Account Amendment- Address Change ask place order']=='Amber Error')&(df_ai_pandas['transcript_x'].str.contains('placing orders', case=False, na=False)),'No Error', df_ai_pandas['Result: Account Amendment- Address Change ask place order'] )

        df_ai_pandas['Result: Account Amendment- Address Change ask place order'] = np.where((df_ai_pandas['Result: Account Amendment- Address Change ask place order']=='Amber Error')&(df_ai_pandas['transcript_x'].str.contains('placing an order', case=False, na=False)),'No Error', df_ai_pandas['Result: Account Amendment- Address Change ask place order'] )

        


        df_ai_pandas['Result: Account Amendment- Address Change ask place order'] = np.where((df_ai_pandas['Result: Account Amendment- Address Change ask place order']=='Amber Error')&(df_ai_pandas['transcript_x'].str.contains('placing any orders', case=False, na=False)),'No Error', df_ai_pandas['Result: Account Amendment- Address Change ask place order'] )

        


  
        df_ai_pandas['Result: Account Amendment- Address Change ask to order after 24hrs'] = np.where(

        ((df_ai_pandas['Did the agent ask caller to allow until tomorrow before placing next order?'].str.lower() == 'no' ) & (df_ai_pandas['Does the conversation involve address change?'].str.lower() == 'yes' ) & (df_ai_pandas["Was there any discussion about order being placed/re-ordered/cancelled/processed?"].str.lower() == 'no') & (df_ai_pandas['Was there any conversation regarding placing orders/placed orders?'].str.lower() =='no') & (df_ai_pandas['Was there any discussion about any orders?'].str.lower() =='no')
        & (df_ai_pandas['Does the caller wants to place an order?'].str.lower() =='no')& (df_ai_pandas['In the conversation is the address change proccess actually followed/change done during call?'].str.lower() =='yes')
         
         )
            
            ,'Amber Error' , 'No Error')

        df_ai_pandas['Result: Account Amendment- Address Change ask to order after 24hrs'] = np.where(

        ((df_ai_pandas['Does the conversation involve address change?'].str.lower() == 'no' ) |((df_ai_pandas['Level 2'] != 'Change Of Detail - Address' )&(df_ai_pandas['Level 2'] != 'Change Of Address Used Without Consent' )))
            
            ,'N.A.' , df_ai_pandas['Result: Account Amendment- Address Change ask to order after 24hrs'])



        df_ai_pandas['Result: Account Amendment- Address Change mobile access for pin'] = np.where(

        (((df_ai_pandas['Did the agent ask if caller has access to the mobile/phone number?'].str.lower() == 'no') & (df_ai_pandas['Did the agent ask any additional security questions or digits of card?'].str.lower() == 'no') ) & (df_ai_pandas['Does the conversation involve address change?'].str.lower() == 'yes' )
         & (df_ai_pandas['Was any PIN discussed in the conversation?'].str.lower() == 'no') & (df_ai_pandas["Did the conversation involve discussion over caller's registered telephone number/mobile number/any phone?"].str.lower() == 'no') & (df_ai_pandas['In the conversation is the address change proccess actually followed/change done during call?'].str.lower() =='yes')
         
         )
            
            ,'Red Error' , 'No Error')


        df_ai_pandas['Result: Account Amendment- Address Change mobile access for pin'] = np.where(

        ((df_ai_pandas['Does the conversation involve address change?'].str.lower() == 'no' ) |((df_ai_pandas['Level 2'] != 'Change Of Detail - Address' )&(df_ai_pandas['Level 2'] != 'Change Of Address Used Without Consent' )))
            
            ,'N.A.' , df_ai_pandas['Result: Account Amendment- Address Change mobile access for pin'])
        
        df_ai_pandas['Result: Account Amendment- Address Change mobile access for pin'] = np.where((df_ai_pandas['Result: Account Amendment- Address Change mobile access for pin']=='Red Error')&(df_ai_pandas['transcript_x'].str.contains('pin', case=False, na=False, regex=True)),'No Error', df_ai_pandas['Result: Account Amendment- Address Change mobile access for pin'] )

        df_ai_pandas['Result: Account Amendment- Address Change mobile access for pin'] = np.where((df_ai_pandas['Result: Account Amendment- Address Change mobile access for pin']=='Red Error')&(df_ai_pandas['transcript_x'].str.contains('4 digit pin', case=False, na=False, regex=True)),'No Error', df_ai_pandas['Result: Account Amendment- Address Change mobile access for pin'] )

        df_ai_pandas['Result: Account Amendment- Address Change mobile access for pin'] = np.where((df_ai_pandas['Result: Account Amendment- Address Change mobile access for pin']=='Red Error')&(df_ai_pandas['transcript_x'].str.contains('four digit pin', case=False,na=False, regex=True)),'No Error', df_ai_pandas['Result: Account Amendment- Address Change mobile access for pin'] )

        df_ai_pandas['Result: Account Amendment- Address Change mobile access for pin'] = np.where((df_ai_pandas['Result: Account Amendment- Address Change mobile access for pin']=='Red Error')&(df_ai_pandas['transcript_x'].str.contains('number ending', case=False, na=False, regex=True)),'No Error', df_ai_pandas['Result: Account Amendment- Address Change mobile access for pin'] )

        df_ai_pandas['Result: Account Amendment- Address Change mobile access for pin'] = np.where((df_ai_pandas['Result: Account Amendment- Address Change mobile access for pin']=='Red Error')&(df_ai_pandas['transcript_x'].str.contains('phone number', case=False, na=False, regex=True)),'No Error', df_ai_pandas['Result: Account Amendment- Address Change mobile access for pin'] )

        df_ai_pandas['Result: Account Amendment- Address Change mobile access for pin'] = np.where((df_ai_pandas['Result: Account Amendment- Address Change mobile access for pin']=='Red Error')&(df_ai_pandas['transcript_x'].str.contains('cell number', case=False, na=False, regex=True)),'No Error', df_ai_pandas['Result: Account Amendment- Address Change mobile access for pin'] )

        df_ai_pandas['Result: Account Amendment- Address Change mobile access for pin'] = np.where((df_ai_pandas['Result: Account Amendment- Address Change mobile access for pin']=='Red Error')&(df_ai_pandas['transcript_x'].str.contains('mobile number', case=False, na=False, regex=True)),'No Error', df_ai_pandas['Result: Account Amendment- Address Change mobile access for pin'] )

        df_ai_pandas['Result: Account Amendment- Address Change mobile access for pin'] = np.where((df_ai_pandas['Result: Account Amendment- Address Change mobile access for pin']=='Red Error')&(df_ai_pandas['transcript_x'].str.contains('contact number', case=False, na=False, regex=True)),'No Error', df_ai_pandas['Result: Account Amendment- Address Change mobile access for pin'] )
        
        #changes for Pay in 3
        df_ai_pandas['Result: Account Amendment- Address Change mobile access for pin'] = np.where((df_ai_pandas['AccountType'] =='PayIn3'), 'No Error', df_ai_pandas['Result: Account Amendment- Address Change mobile access for pin'] )

         #cash account
        df_ai_pandas['Result: Account Amendment- Address Change mobile access for pin'] = np.where((df_ai_pandas['account_type'].str.lower() == 'cash' ), 'No Error', df_ai_pandas['Result: Account Amendment- Address Change mobile access for pin'])





        df_ai_pandas['Result: Account Amendment- Address Change pin fail security question'] = np.where(

        ((df_ai_pandas["Did the pin verification through phone fail?"].str.lower() == 'yes' )&(df_ai_pandas["If the pin verification fail, were the security questions asked?"].str.lower() == 'no' ) & (df_ai_pandas['Does the conversation involve address change?'].str.lower() == 'yes' ) & (df_ai_pandas['In the conversation is the address change proccess actually followed/change done during call?'].str.lower() =='yes') &(df_ai_pandas["Did the agent mention that the pin verification fail?"].str.lower() =='yes')
         
         )
            
            ,'Red Error' , 'No Error')




        df_ai_pandas['Result: Account Amendment- Address Change pin fail security question'] = np.where(

        ((df_ai_pandas['Does the conversation involve address change?'].str.lower() == 'no' )|((df_ai_pandas['Level 2'] != 'Change Of Detail - Address' )&(df_ai_pandas['Level 2'] != 'Change Of Address Used Without Consent' )))
            
            ,'N.A.' , df_ai_pandas['Result: Account Amendment- Address Change pin fail security question'])
        
        #changes for Pay in 3
        df_ai_pandas['Result: Account Amendment- Address Change pin fail security question'] = np.where((df_ai_pandas['AccountType'] =='PayIn3'), 'No Error', df_ai_pandas['Result: Account Amendment- Address Change pin fail security question'] )

        #cash account
        df_ai_pandas['Result: Account Amendment- Address Change pin fail security question'] = np.where((df_ai_pandas['account_type'].str.lower() == 'cash' ), 'No Error', df_ai_pandas['Result: Account Amendment- Address Change pin fail security question'])




        df_ai_pandas['Result: Account Amendment- Address Change pin fail security question'] = np.where((df_ai_pandas['Result: Account Amendment- Address Change pin fail security question']=='Red Error')&(df_ai_pandas['transcript_x'].str.contains('pin', case=False, na=False, regex=True)),'No Error', df_ai_pandas['Result: Account Amendment- Address Change pin fail security question'] )

        df_ai_pandas['pin present'] =df_ai_pandas['transcript_x'].str.contains('pin', case=False, na=False, regex=True)


        # in development  sub parameters

        # df_ai_pandas['Result: Account Amendment- Address Change ask for contact number change'] = np.where(

        # ((df_ai_pandas['Did the agent ask the caller if contact number needs to be changed?'].str.lower() == 'no' ) & (df_ai_pandas['Does the conversation involve address change?'].str.lower() == 'yes' ))
            
        #     ,'Amber Error' , 'No Error')



        # df_ai_pandas['Result: Account Amendment- Address Change ask for contact number change'] = np.where(

        # ((df_ai_pandas['Does the conversation involve address change?'].str.lower() == 'no' )|((df_ai_pandas['Level 2'] != 'Change Of Detail - Address' )&(df_ai_pandas['Level 2'] != 'Change Of Address Used Without Consent' )))
            
        #     ,'N.A.' , df_ai_pandas['Result: Account Amendment- Address Change ask for contact number change'])



        # df_ai_pandas['Result: Account Amendment- Address Change ask for email change'] = np.where(

        # ((df_ai_pandas['Did the agent ask the caller if email address needs to be changed?'].str.lower() == 'no' ) & (df_ai_pandas['Does the conversation involve address change?'].str.lower() == 'yes' ))
            
        #     ,'Amber Error' , 'No Error')


        # df_ai_pandas['Result: Account Amendment- Address Change ask for email change'] = np.where(

        # ((df_ai_pandas['Does the conversation involve address change?'].str.lower() == 'no' )|((df_ai_pandas['Level 2'] != 'Change Of Detail - Address' )&(df_ai_pandas['Level 2'] != 'Change Of Address Used Without Consent' )))
            
        #     ,'N.A.' , df_ai_pandas['Result: Account Amendment- Address Change ask for email change'])



        df_ai_pandas['Result: Account Amendment- Address Change address change confirmation'] = np.where(

        ((df_ai_pandas['Did the agent confirm address change and that a letter will be sent to new address?'].str.lower() == 'no' ) & (df_ai_pandas['Does the conversation involve address change?'].str.lower() == 'yes' ) & (df_ai_pandas['In the conversation is the address change proccess actually followed/change done during call?'].str.lower() =='yes')
         
         )
            
            ,'Amber Error' , 'No Error')


        df_ai_pandas['Result: Account Amendment- Address Change address change confirmation'] = np.where(

        ((df_ai_pandas['Does the conversation involve address change?'].str.lower() == 'no' )|((df_ai_pandas['Level 2'] != 'Change Of Detail - Address' )&(df_ai_pandas['Level 2'] != 'Change Of Address Used Without Consent' )))
            
            ,'N.A.' , df_ai_pandas['Result: Account Amendment- Address Change address change confirmation'])
            



        result_cols = [col for col in df_ai_pandas.columns if col.startswith("Result:") ]        
        df_ai_pandas['Result: Account Amendment- Address Amendment'] =  df_ai_pandas[result_cols].apply(get_result_priority, axis = 1)
        df_ai_pandas['Score: Account Amendment- Address Amendment'] = df_ai_pandas['Result: Account Amendment- Address Amendment'].map(score_map)


        df_ai_pandas['Result: Account Amendment- Address Amendment'] = np.where(

        ((df_ai_pandas['Does the conversation involve address change?'].str.lower() == 'no' ))
            
            ,'N.A.' , df_ai_pandas['Result: Account Amendment- Address Amendment'])

        df_ai_pandas['Result: Account Amendment- Address Amendment'] = np.where(

        ((df_ai_pandas['Level 2'] == 'Change Of Detail - Address' )|(df_ai_pandas['Level 2'] == 'Change Of Address Used Without Consent' )  )
            
            ,df_ai_pandas['Result: Account Amendment- Address Amendment'] , 'N.A.')
            
        df_ai_pandas['Score: Account Amendment- Address Amendment'] = np.where(
            df_ai_pandas['Result: Account Amendment- Address Amendment'] == "N.A." , 10, df_ai_pandas['Score: Account Amendment- Address Amendment'])	



        ######################### change of email #####################################
        # "Did the conversation have anything about pin verification failing/did pin verification fail?",
        #     "Did the email change happen during the conversation/transcipt?",

        df_ai_pandas['Result: Account Amendment- Email Change Phone handy for pin'] = np.where(

        ((df_ai_pandas['Did the agent ask about the phone being accessible for pin verification?'].str.lower() == 'no' ) & (df_ai_pandas['Does the conversation involve email change?'].str.lower() == 'yes' )&(df_ai_pandas['Did the email change happen during the conversation/transcipt?'].str.lower() == 'yes'))
            
            ,'Amber Error' , 'No Error')


        df_ai_pandas['Result: Account Amendment- Email Change Phone handy for pin'] = np.where(

        ((df_ai_pandas['Does the conversation involve email change?'].str.lower() == 'no' ) | (df_ai_pandas['Level 2'] != 'Change Of Detail - Email Address' ) )
            
            ,'N.A.' , df_ai_pandas['Result: Account Amendment- Email Change Phone handy for pin'])
        
         #changes for Pay in 3
        df_ai_pandas['Result: Account Amendment- Email Change Phone handy for pin'] = np.where((df_ai_pandas['AccountType'] =='PayIn3'), 'No Error', df_ai_pandas['Result: Account Amendment- Email Change Phone handy for pin'] )


        df_ai_pandas['Result: Account Amendment- Email Change Pin verified'] = np.where(

        ((df_ai_pandas['Did the pin verification through phone happen for email change?'].str.lower() == 'no' ) & (df_ai_pandas['Does the conversation involve email change?'].str.lower() == 'yes' )&(df_ai_pandas['Did the email change happen during the conversation/transcipt?'].str.lower() == 'yes'))
            
            ,'Amber Error' , 'No Error')


        df_ai_pandas['Result: Account Amendment- Email Change Pin verified'] = np.where(

        ((df_ai_pandas['Does the conversation involve email change?'].str.lower() == 'no' ) | (df_ai_pandas['Level 2'] != 'Change Of Detail - Email Address' ))
            
            ,'N.A.' , df_ai_pandas['Result: Account Amendment- Email Change Pin verified'])
        
         #changes for Pay in 3
        df_ai_pandas['Result: Account Amendment- Email Change Pin verified'] = np.where((df_ai_pandas['AccountType'] =='PayIn3'), 'No Error', df_ai_pandas['Result: Account Amendment- Email Change Pin verified'] )


        df_ai_pandas['Result: Account Amendment- Email Change Pin fail security'] = np.where(

        ((df_ai_pandas['If the pin verification fail, did the agent ask security questions?'].str.lower() == 'no' ) & (df_ai_pandas['Does the conversation involve email change?'].str.lower() == 'yes' )& (df_ai_pandas['Did the conversation have anything about pin verification failing/did pin verification fail?'].str.lower() == 'yes') & (df_ai_pandas['Did the email change happen during the conversation/transcipt?'].str.lower() == 'yes'))
            
            ,'Amber Error' , 'No Error')


        df_ai_pandas['Result: Account Amendment- Email Change Pin fail security'] = np.where(

        ((df_ai_pandas['Does the conversation involve email change?'].str.lower() == 'no' ) | (df_ai_pandas['Level 2'] != 'Change Of Detail - Email Address' ))
            
            ,'N.A.' , df_ai_pandas['Result: Account Amendment- Email Change Pin fail security'])
        
         #changes for Pay in 3
        df_ai_pandas['Result: Account Amendment- Email Change Pin fail security'] = np.where((df_ai_pandas['AccountType'] =='PayIn3'), 'No Error', df_ai_pandas['Result: Account Amendment- Email Change Pin fail security'] )


        df_ai_pandas['Result: Account Amendment- Email Change Confirm'] = np.where(

        ((df_ai_pandas['After pin verification, did agent confirm email change?'].str.lower() == 'no' ) & (df_ai_pandas['Does the conversation involve email change?'].str.lower() == 'yes' )& (df_ai_pandas['Did the email change happen during the conversation/transcipt?'].str.lower() == 'yes'))
            
            ,'Compliant with development' , 'Good Customer Outcome')


        df_ai_pandas['Result: Account Amendment- Email Change Confirm'] = np.where(

        ((df_ai_pandas['Does the conversation involve email change?'].str.lower() == 'no' ) | (df_ai_pandas['Level 2'] != 'Change Of Detail - Email Address' ))
            
            ,'N.A.' , df_ai_pandas['Result: Account Amendment- Email Change Confirm'])



        result_cols = [col for col in df_ai_pandas.columns if col.startswith("Result:") ]        
        df_ai_pandas['Result: Account Amendment- Email Amendment'] = df_ai_pandas[result_cols].apply(get_result_priority, axis = 1)
        df_ai_pandas['Score: Account Amendment- Email Amendment'] = df_ai_pandas['Result: Account Amendment- Email Amendment'].map(score_map)

        df_ai_pandas['Result: Account Amendment- Email Amendment'] = np.where(

        ((df_ai_pandas['Does the conversation involve email change?'].str.lower() == 'no' ))
            
            ,'N.A.' , df_ai_pandas['Result: Account Amendment- Email Amendment'])


        df_ai_pandas['Result: Account Amendment- Email Amendment'] = np.where(

        ((df_ai_pandas['Level 2'] == 'Change Of Detail - Email Address' )  )
            
            ,df_ai_pandas['Result: Account Amendment- Email Amendment'] , 'N.A.')
            
        df_ai_pandas['Score: Account Amendment- Email Amendment'] = np.where(
            df_ai_pandas['Result: Account Amendment- Email Amendment'] == "N.A." , 10, df_ai_pandas['Score: Account Amendment- Email Amendment'])	

        df_ai_pandas['Result: Gone away set'] = np.where((df_ai_pandas['Level 2'] == 'Gone Away Set') & (df_ai_pandas['HoldOrderIndicator'].str.lower() != 'g'), 'Red Error', 'No Error')
        df_ai_pandas['Result: Gone away set'] = np.where(df_ai_pandas['Level 2'].isin(['Gone Away Set']),df_ai_pandas['Result: Gone away set'], 'N.A.')
        final_result_cols = ['Result: Gone away set','Result: Account Amendment- Email Amendment','Result: Account Amendment- Address Amendment','Result: Account Closure','Result: Account Amendment- Name Amendment', 'Result: Account Amendment- Phone Amendment' ]
        df_ai_pandas['iAudit Result: Account Amendment and Closure']= df_ai_pandas[final_result_cols].apply(get_result_priority, axis = 1)
        level_2_req_list = level_2_req_list + ['Gone Away Set']
        df_ai_pandas['iAudit Result: Account Amendment and Closure']= np.where(df_ai_pandas['Level 2'].isin(level_2_req_list), df_ai_pandas['iAudit Result: Account Amendment and Closure'], 'N.A.' )
        df_ai_pandas['iAudit Result: Account Amendment and Closure']=np.where(df_ai_pandas["agent_order_max"]== df_ai_pandas["agent_order"],df_ai_pandas['iAudit Result: Account Amendment and Closure'], 'N.A.' )
        # df_ai_pandas['iAudit Result: Account Amendment and Closure']=np.where(
        #                                             ((df_ai_pandas["Output_final"]=="") |(df_ai_pandas["Output_final"].isna()))
                                                    
        #                                             , 'N.A.',df_ai_pandas['iAudit Result: Account Amendment and Closure'] 
        #     )
        result_cols = [col for col in df_ai_pandas.columns if col.startswith("Result:") ]
        result_cols = list(set(result_cols)- set(final_result_cols)) 
        error_list = ["red error", "amber error", "compliant with development"]
        df_ai_pandas['Remarks'] = df_ai_pandas.apply(lambda row: ', '.join([f"{str(row[col]).strip()}: {col.replace('Result: Account Amendment-', '').strip()}" for col in result_cols if str(row[col]).strip().lower() in error_list])
            , axis =1)
        df_ai_pandas['iAudit Score: Account Amendment and Closure']= df_ai_pandas['iAudit Result: Account Amendment and Closure'].map(score_map)	
        df_ai_pandas['iAudit Remarks: Account Amendment and Closure']=	df_ai_pandas['Remarks']
        remarks_dict = {
        "Phone Change Validation": "Phone Updation: the changed  phone details not same as in DB",
        "Name Change Validation" : "Name Updation: the changed name details not same as in DB",
        "Address Change ask place order" : "Address Updation : Agent didn’t confirm if the customer wanted to place an order today",
        "Address Change ask to order after 24hrs": "Address Updation : Agent didn’t advise the caller to wait 24 hours before placing the next order",
        "Address Change mobile access for pin": "Address Updation : Agent didn’t check if the caller had access to the registered mobile number",
        "Address Change pin fail security question": "Address Updation : Agent didn’t complete PIN verification or follow security question protocol after PIN failure.",
        "Address Change ask for contact number change": "Address Updation : Agent didn’t ask if the contact number needed updating after address change",
        "Address Change ask for email change": "Address Updation : Agent didn’t ask if the email address needed updating after address change",
        "Address Change address change confirmation": "Address Updation : Agent didn’t confirm the address change or inform the customer about the confirmation letter",
        "Email Change Phone handy for pin": "Email Updation : Agent didn’t confirm if the phone was accessible for PIN verification",
        "Email Change Pin verified": "Email Updation : Agent didn’t verify PIN through phone",
        "Email Change Pin fail security": "Email Updation : Agent didn’t ask security questions after the PIN verification failed",
        "Email Change Confirm": "Email Updation : Agent didn’t confirm the email change after successful PIN verification",
        "Account closure reason_(AccountClosure)": "Account Closure : Agent didn’t ask for the reason behind the account closure",
        "Account closure communication_(AccountClosure)": "Account Closure : Agent didn’t confirm if the customer wanted to receive closure communication.",
        "Account Closure DirectDebit": "Account Closure : Agent didn’t confirm if account closure and direct debit cancellation were completed",
        "Pending returns or refund_(AccountClosure)": "Account Closure : Agent didn’t ask about any pending refunds or return items",
        "No pending order to be delivered_(AccountClosure)": "Account Closure : Agent didn’t ask for any pending orders before closure"
        }
        df_ai_pandas['iAudit Remarks: Account Amendment and Closure'] = df_ai_pandas['iAudit Remarks: Account Amendment and Closure'].replace(remarks_dict, regex = True)

        df_ai_pandas['iAudit Remarks: Account Amendment and Closure'] = np.where(df_ai_pandas['iAudit Score: Account Amendment and Closure']==10,"",df_ai_pandas['iAudit Remarks: Account Amendment and Closure'] )


        #changes for SpecialAccountDescription == Staff 

        df_ai_pandas['iAudit Result: Account Amendment and Closure']= np.where(((df_ai_pandas['iAudit Remarks: Account Amendment and Closure'].str.contains("Address Updation",case=False, na=False)) & (df_ai_pandas['SpecialAccountDescription']=='Staff')), 'Good Customer Outcome', df_ai_pandas['iAudit Result: Account Amendment and Closure'] )

        df_ai_pandas['iAudit Score: Account Amendment and Closure']= np.where(((df_ai_pandas['iAudit Remarks: Account Amendment and Closure'].str.contains("Address Updation",case=False, na=False)) & (df_ai_pandas['SpecialAccountDescription']=='Staff')), 10, df_ai_pandas['iAudit Score: Account Amendment and Closure'] )

        df_ai_pandas['iAudit Remarks: Account Amendment and Closure']= np.where(((df_ai_pandas['iAudit Remarks: Account Amendment and Closure'].str.contains("Address Updation",case=False, na=False)) & (df_ai_pandas['SpecialAccountDescription']=='Staff')), "", df_ai_pandas['iAudit Remarks: Account Amendment and Closure'] )

        df_ai_pandas['iAudit Confidence: Account Amendment and Closure']=np.where((df_ai_pandas['iAudit Remarks: Account Amendment and Closure'].str.contains("Name Updation: the changed name details not same as in DB",case=False, na=False))|(df_ai_pandas['iAudit Remarks: Account Amendment and Closure'].str.contains("Phone Updation: the changed  phone details not same as in DB",case=False, na=False)), 'Low', 'High')

         #Partial Transcripts
        df_ai_pandas['iAudit Result: Account Amendment and Closure']= np.where(( (df_ai_pandas['Is the conversation in the transcript partial/ended abruptly/incomplete conversation?'].str.lower()== 'yes')| ( df_ai_pandas["Did the agent ask if they have resolved and said bye?"].str.lower()== 'no')), 'Good Customer Outcome',df_ai_pandas['iAudit Result: Account Amendment and Closure'] )

        df_ai_pandas['iAudit Score: Account Amendment and Closure']= np.where( (df_ai_pandas['Is the conversation in the transcript partial/ended abruptly/incomplete conversation?'].str.lower()== 'yes')| ( df_ai_pandas["Did the agent ask if they have resolved and said bye?"].str.lower()== 'no'), 10,df_ai_pandas['iAudit Score: Account Amendment and Closure'] )

        df_ai_pandas['iAudit Remarks: Account Amendment and Closure']= np.where( (df_ai_pandas['Is the conversation in the transcript partial/ended abruptly/incomplete conversation?'].str.lower()== 'yes')| ( df_ai_pandas["Did the agent ask if they have resolved and said bye?"].str.lower()== 'no'), "",df_ai_pandas['iAudit Remarks: Account Amendment and Closure'] )

        df_ai_pandas['iAudit Confidence: Account Amendment and Closure']= np.where(  (df_ai_pandas['Is the conversation in the transcript partial/ended abruptly/incomplete conversation?'].str.lower()== 'yes')| ( df_ai_pandas["Did the agent ask if they have resolved and said bye?"].str.lower()== 'no'), "High",df_ai_pandas['iAudit Confidence: Account Amendment and Closure'] )


        df_ai_pandas = df_ai_pandas.applymap(remove_illegal_chars)

        check_account_closure = ['Close Account - No Reason' ,'Close Account - Service','Change Of Detail - Address','Change Of Address Used Without Consent', 'Change Of Detail - Telephone Number','Change Of Detail - Email Address','Change Of Detail - Name Or Title', 'Close Account - No Reason' ,'Close Account - Service','Gone Away Set']
        df_ai_pandas['iAudit Result: Account Amendment and Closure'] = np.where((df_ai_pandas['iAudit Result: Account Amendment and Closure'] =='N.A.') & (df_ai_pandas['Level 2'].isin(check_account_closure)),'Good Customer Outcome',df_ai_pandas['iAudit Result: Account Amendment and Closure'])

        df_ai_pandas.to_excel("AccountAmendemntIntermediateResultsAfterChange.xlsx")

        #df_ai_pandas.to_excel("AccountAmendemntIntermediateResultsAfterChange.xlsx")
        df_ai_pandas_filtered = df_ai_pandas[['callid_agentid','callid','AgentId','Level 2','iAudit Result: Account Amendment and Closure','iAudit Score: Account Amendment and Closure','iAudit Remarks: Account Amendment and Closure', 'iAudit Confidence: Account Amendment and Closure']]
        replace_map_result ={ 'Amber Error': 'Non-Compliant No Poor Outcome',
                      'Red Error': 'Non-Compliant Poor Outcome',
                      'Compliant with development': 'Compliant with Development',
                      'Good Customer Outcome': 'Good Customer Outcome',
                      'N.A.': 'N.A.'
                     

        }
        df_ai_pandas_filtered['iAudit Result: Account Amendment and Closure'] = df_ai_pandas_filtered['iAudit Result: Account Amendment and Closure'].replace(replace_map_result).replace(['', None, np.nan], 'N.A.')
        df_ai_pandas['iAudit Score: Account Amendment and Closure']= np.where(
            df_ai_pandas_filtered['iAudit Result: Account Amendment and Closure'].str.contains('N.A.'),10, df_ai_pandas_filtered['iAudit Score: Account Amendment and Closure']
            )
       

        df_ai_pandas_filtered.to_excel(result_path)

    except Exception as e:
        log.error(f"Module name: Account Amendment and Closure- Error: {e}", exc_info=True)    
  