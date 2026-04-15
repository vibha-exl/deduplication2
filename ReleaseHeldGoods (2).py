# !pip install json-repair -q
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
from iaudit_logger import get_logger
from ReleaseHeldGoodsPrompts import *
from pyspark.sql.functions import format_string
from pyspark.sql.functions import array, lit, when, size, concat_ws
from pyspark.sql import functions as F
import yaml
log = get_logger()


warnings.filterwarnings("ignore")
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

if __name__=="__main__":
    formatted_date ='2026-04-06'
    # transcripts = pd.read_excel(r'/Workspace/Users/ananthagiri_abhiram@next.co.uk/iAudit_Trisha/ReleaseHeldGoodsTranscripts.xlsx')
    # transcripts = pd.read_excel(r'/Workspace/Users/ananthagiri_abhiram@next.co.uk/iAudit_Trisha/UATpool2CallsRedcatedTranscripts.xlsx')
    # transcripts = pd.read_excel(r'/Workspace/Users/ananthagiri_abhiram@next.co.uk/iAudit_Trisha/AvailableTranscriptsQCAaudited2025.xlsx')

#     transcripts = transcripts[transcripts['callid'].isin([ "006d50e5-0ae8-4a2c-90c4-db07e388b4ee",
# "023ab213-95ca-4b38-be50-b0d27d765e9c",
# "03d7a489-9631-4055-8d99-ab2351597cc4",
# "04af1bee-1b72-42cb-9060-1d450a0e89e7",
# "059b7bf6-ccb4-4c07-9eac-d9e72bba873a",
# "0751b5c2-30e6-4e08-9ec1-bc29878401ae",
# "087e868a-a40a-41c1-ae76-c59610f1234d",
# "098206de-26f0-4a69-91bc-bf5838451459",
# "10202270-a29a-4d36-ad00-d353f814d858",
# "11d53068-9612-4006-a1e2-46bf6f3c2085",
# "15356034-77e7-4417-9e45-cd0a0d24b8e8",
# "20c3a3f8-dbb9-4ed3-ab00-d6e4e1ed7e85",
# "2741a817-defe-41d2-a6c5-b1151202da33"

#       ])]
   
    raw_table_query = f"""SELECT * FROM contactcentre_prod.transcripts.transcripts_raw WHERE DATE(createdate) = '{formatted_date}'"""
   
    full_raw_table = spark.sql(raw_table_query)
    transcripts = full_raw_table.toPandas()
    transcripts =transcripts.rename(columns={'AgentID':'AgentId'})
    transcripts =transcripts.rename(columns={'Callid':'callid'})
    print(transcripts.columns)

    num_unique_callid_agentid = transcripts[['callid', 'AgentId']].drop_duplicates().shape[0]
    print("num_unique_callid_agentid transcript", num_unique_callid_agentid)
    print("transcripts", transcripts.shape)

    

    total_calls = len(transcripts['callid'].unique().tolist())
    filtered_calls = transcripts['callid'].unique().tolist()[:total_calls]
    #filtered_calls = transcripts['callid'].unique().tolist()[:100]
    df_transcripts = transcripts[transcripts['callid'].isin(filtered_calls)].reset_index(drop=True)
    print("df_transcripts", df_transcripts.shape)

    audit_callids = df_transcripts['callid'].unique().tolist()

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
    print("Logger unique callids", len(df_logger['CallId'].unique().tolist()))
    customer_account_number_list = df_logger['account_number'].unique().tolist()
    customer_account_number_list = [x for x in customer_account_number_list if str(x) != 'nan']
    print("unique_account_number",len(customer_account_number_list))
    customer_account_number_list = [ x for x in customer_account_number_list if x is not None]
    print("Not none unique_account_number",len(customer_account_number_list))

    spark_vw_payments = spark.sql(f"""
            SELECT *
            FROM `contactcentre_prod`.`iaudit`.`vw_payments`
            WHERE `CustomerNo` IN {tuple(customer_account_number_list)}      
        """)
    df_vw_payments= spark_vw_payments.toPandas()
    df_vw_payments = df_vw_payments.rename(columns={'CustomerNo': 'account_number'})  
    df_vw_payments = df_vw_payments.drop_duplicates(subset=['account_number'], keep='first')
    list_of_account_number_in_payment_table = df_vw_payments['account_number'].unique().tolist()
    print("unique_account_number in payment table",len(list_of_account_number_in_payment_table))

    level_2_req_list = ['Store Referral - Authorised' ,'Store Referral - Declined' ] 
    df_logger['priority'] =df_logger['Level 2'].isin(level_2_req_list).astype(int)
    print("Logger before",df_logger.shape)
    df_logger_unique = (df_logger.sort_values(['CallId', 'priority'], ascending=[True, False]).drop_duplicates(subset = 'CallId', keep='first').drop(columns='priority'))
    print(df_logger_unique.shape)

    data = pd.merge(df_transcripts, df_logger_unique, left_on='callid', right_on='CallId', how='left')
    data['callid_agentid']= data['callid'] + "|" + data['AgentId']
    print("data",data.shape)
    grouped_transcript_all = data.groupby('callid_agentid').apply(lambda x: "\n".join(f"{row['channel']}: {row['transcript']}" for _, row in x.iterrows())).reset_index(name = 'transcript')
    print("Grouped T", grouped_transcript_all.shape)
    data_intermediate = data.groupby('callid_agentid', as_index = False).first()
    data_req = pd.merge(grouped_transcript_all, data_intermediate, on='callid_agentid', how='left')
    data_req['createdate'] = pd.to_datetime(data_req['createdate'], format = 'mixed', errors='coerce')
    data_req = data_req.sort_values(by=["callid", 'createdate'], ascending =[True, True])
    data_req['createdate'] = data_req['createdate'].astype(str)
    data_req['agent_order'] = data_req.groupby("callid").cumcount()+1
    data_req['agent_order_max'] = data_req.groupby("callid")["agent_order"].transform("max")
    data_req['payment_ever_done']=  np.where(data_req['account_number'].isin(list_of_account_number_in_payment_table), "Yes", "No")
    print("final data",data_req.shape)

    #filters
    data_req['eligibility'] = np.where(
        ((data_req['Level 2'].isin(level_2_req_list)) & (data_req['account_type'].str.lower()== 'credit') & (data_req['Queues'].isin(['Voice Credit Referral'])) ), 1, 0


    )
    log.info(f"Final data shape {data_req.shape}")

    df_prompts = (
                    spark.createDataFrame(data_req)
                    .withColumn( "user_prompt", F.when((F.col("eligibility")==1) , F.lit(user_prompt_release_held_goods))
                                .otherwise(F.lit(""))
                                
                                )
                    .withColumn("final_prompt", F.when( (F.col("eligibility")==1),
                                                F.concat_ws( "\n\n",
                                                            F.lit("SYSTEM PROMPT:"),
                                                            F.lit(system_prompt),
                                                            F.lit("USER PROMPT:"),
                                                            F.col("user_prompt"),
                                                            F.lit("TRANSCRIPT:"),
                                                            F.col("transcript_x"))
    
                                                             )
                                .otherwise(F.lit(""))
                                )
                    )
    
    df_ai = (
                df_prompts.withColumn(
                    "ai_response", F.when(
                        (F.col("eligibility")==1) &
                    (F.col("agent_order_max")== F.col("agent_order")) &
                    (F.col("user_prompt") != ""), F.expr(
                        """ai_query('contact_centre_internal_batch',request => final_prompt)"""
                            )
                    
                )
            )
            )
    #display(df_ai.toPandas())
    df_ai.toPandas().to_excel("ReleseHeldGoodsIntermediateResultsv1.xlsx")


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
        "Is the call from a store about authorization code/store card referral?",
        "Did the agent confirm the full name and/or customer number?",
        "Did the agent ask for value of sale?",
        "Did the agent ask for the store number/ID?",
        "Did the agent ask if the customer is using their virtual card through the Next Pay App?",
        "Did the agent ask if the customer is using a physical card?",
        "Was the customer using a physical card?",
        "Did the agent ask to confirm the app and not a screenshot?",
        "Did the store consultant confirm that customer is registered in app?",
        "Did the agent decline the sale if due to no screenshot/ app registeration not confirmed by store consultant?",
        "Did the agent ensured from store consulatant that not on loudspeaker and to answer discreetly?",
        "Did the agent ask the consultant if the order contains a Gift Card?",
        "Did the store consultant deny customer having the gift card in order?",
        "Did the agent decline sale if gift card in order?",
        "Did the agent ask if the customer using their store card as payment for retail order?",
        "Did the agent ask if there an electrical item in the order or branded items over value £100?",
        "Did the agent ask if there are multiple electrical items in the order or branded items over value £100?",
        "Did the agent ask if the order is for large furniture items only?",
        "Did the order have only large furniture item?",
        "Did the agent authorize the transcation if it has only furniture items?",
        "Did the agent ask if the order value exceeds 500 £?",
        "Did the order also have fashion items?",
        "Is the value of fashion item over 500 £?",
        "Did the agent ask about the registered mobile number which is accessible?",
        "Is the number accessible to customer?",
        "Did the agent ask for pin verification if the mobile number is apt?",
        "If the pin is wrong/not confirmed then did agent decline sale?",
        "Did the agent ask for email for pin verification?",
        "Was the email accessible to the customer?",
        "Did the agent did a pin confirmation via email successfully?",
        "Did the agent remind the store consultant to get customer to sign till receipt?",
        "Did the agent authorize the transaction?",
        "Did the agent decline the transaction?",
        "Evidence",
        "Is the customer using the store card as payment for retail order?",
        "Is there an electrical item in the order or branded items over value £100?",
        "Is the order for large furniture items only?",
        "Is the order value exceeds 500?",
        "Was transaction/authorization declined due to credit limit/asked to talk to credit team?",
         "Is the conversation in the transcript partial/ended abruptly/incomplete conversation?",
         "Did the agent ask if they have resolved and said bye?"
      
                 ]
    for col in ai_response_column_list:
            if col not in df_ai_pandas.columns:
                df_ai_pandas[col] = "N.A."

    df_ai_pandas[ai_response_column_list] = df_ai_pandas[ai_response_column_list].replace("", np.nan).fillna("N.A.")          


    df_ai_pandas['iAudit Result: Release Held Goods']= "N.A."
    df_ai_pandas['iAudit Score: Release Held Goods']= 10	
    df_ai_pandas['iAudit Remarks: Release Held Goods']=	""


    df_ai_pandas.to_excel("ReleaseHledGoodsAfterInitialisation.xlsx")

    print("Final Scoring happening")
    df_ai_pandas['Result: Release Held Goods-DPA Check'] = np.where(
    (df_ai_pandas['Is the call from a store about authorization code/store card referral?'].str.lower() == 'yes' ) &

    (
        (df_ai_pandas['Did the agent confirm the full name and/or customer number?'].str.lower() == 'no' ) | (df_ai_pandas['Did the agent ask for value of sale?'].str.lower() == 'no' )| (df_ai_pandas['Did the agent ask for the store number/ID?'].str.lower() == 'no' )
        
        )
        
        ,'Red Error' , 'No Error')
    
    df_ai_pandas['Result: Release Held Goods-DPA Check'] = np.where((df_ai_pandas['Result: Release Held Goods-DPA Check']=='Red Error')&(df_ai_pandas['transcript_x'].str.contains('customer number', case=False, na=False))&(df_ai_pandas['transcript_x'].str.contains('value', case=False, na=False))&(df_ai_pandas['transcript_x'].str.contains('store number', case=False, na=False)),'No Error', df_ai_pandas['Result: Release Held Goods-DPA Check'] )

    df_ai_pandas['Result: Release Held Goods-DPA Check'] = np.where((df_ai_pandas['Result: Release Held Goods-DPA Check']=='Red Error')&(df_ai_pandas['transcript_x'].str.contains('customer number', case=False, na=False))&(df_ai_pandas['transcript_x'].str.contains('value', case=False, na=False))&(df_ai_pandas['transcript_x'].str.contains('store ID', case=False, na=False)),'No Error', df_ai_pandas['Result: Release Held Goods-DPA Check'] )
    
    
      
        

    # df_ai_pandas['Result: Release Held Goods- Fraud Check if Virtual Card'] = np.where((df_ai_pandas['Is the call from a store about authorization code/store card referral?'].str.lower() == 'yes' ) &

    # ((df_ai_pandas['Did the agent ask if the customer is using their virtual card through the Next Pay App?'].str.lower() == 'no' ) & (df_ai_pandas['Did the agent ask if the customer is using a physical card?'].str.lower() == 'no') )
        
    #     ,'Amber Error' , 'No Error')


    # df_ai_pandas['Result: Release Held Goods- Fraud Check Virtual Card Screenshot'] = np.where((df_ai_pandas['Is the call from a store about authorization code/store card referral?'].str.lower() == 'yes' ) &

    # ((df_ai_pandas['Did the agent ask to confirm the app and not a screenshot?'].str.lower() == 'no' ) & (df_ai_pandas["Was the customer using a physical card?"].str.lower() == 'no') )
        
    #     ,'Amber Error' , 'No Error')

    df_ai_pandas['Result: Release Held Goods- Gift card'] = np.where(
        (df_ai_pandas['Is the call from a store about authorization code/store card referral?'].str.lower() == 'yes' ) &

    ((df_ai_pandas['Did the agent ask the consultant if the order contains a Gift Card?'].str.lower() == 'no' ) )
        
        ,'Amber Error' , 'No Error')
    
    df_ai_pandas['Result: Release Held Goods- Gift card'] = np.where((df_ai_pandas['Result: Release Held Goods- Gift card']=='Amber Error')&(df_ai_pandas['transcript_x'].str.contains('gift card', case=False, na=False)),'No Error', df_ai_pandas['Result: Release Held Goods- Gift card'] )

    df_ai_pandas['Result: Release Held Goods- Store Card'] = np.where(
        (df_ai_pandas['Is the call from a store about authorization code/store card referral?'].str.lower() == 'yes' ) &

    ((df_ai_pandas['Did the agent ask if the customer using their store card as payment for retail order?'].str.lower() == 'no' ) )
        
        ,'Amber Error' , 'No Error')
    
    df_ai_pandas['Result: Release Held Goods- Store Card'] = np.where((df_ai_pandas['Result: Release Held Goods- Store Card']=='Amber Error')&(df_ai_pandas['transcript_x'].str.contains('retail', case=False, na=False)),'No Error', df_ai_pandas['Result: Release Held Goods- Store Card'] )
    

    df_ai_pandas['Result: Release Held Goods-  Electric Good'] = np.where( (df_ai_pandas['Is the call from a store about authorization code/store card referral?'].str.lower() == 'yes' ) &(df_ai_pandas['Is the customer using the store card as payment for retail order?'].str.lower() == 'yes' ) &

    ((df_ai_pandas['Did the agent ask if there an electrical item in the order or branded items over value £100?'].str.lower() == 'no' ) )
        
        ,'Amber Error' , 'No Error')
    
    df_ai_pandas['Result: Release Held Goods-  Electric Good'] = np.where((df_ai_pandas['Result: Release Held Goods-  Electric Good']=='Amber Error')&(df_ai_pandas['transcript_x'].str.contains('electric', case=False, na=False)),'No Error', df_ai_pandas['Result: Release Held Goods-  Electric Good'] )

    df_ai_pandas['Result: Release Held Goods-  Electric Good'] = np.where((df_ai_pandas['Result: Release Held Goods-  Electric Good']=='Amber Error')&(df_ai_pandas['transcript_x'].str.contains('electrical', case=False, na=False)),'No Error', df_ai_pandas['Result: Release Held Goods-  Electric Good'] )

    # df_ai_pandas['Result: Release Held Goods- Established Account'] = np.where(

    # ((df_ai_pandas['Did the agent decline the transaction?'].str.lower() == 'no' ) & (df_ai_pandas['Established account'] == 'False') )
        
    #     ,'Amber Error' , 'No Error')
        

    df_ai_pandas['Result: Release Held Goods- Multiple Electrical Goods'] = np.where(
        (df_ai_pandas['Is the call from a store about authorization code/store card referral?'].str.lower() == 'yes' ) & (df_ai_pandas['Is the customer using the store card as payment for retail order?'].str.lower() == 'yes' ) & (df_ai_pandas["Is there an electrical item in the order or branded items over value £100?"].str.lower() == 'yes' ) &(df_ai_pandas['payment_ever_done'].str.lower()=='yes') &

    ((df_ai_pandas['Did the agent ask if there are multiple electrical items in the order or branded items over value £100?'].str.lower() == 'no' ) )
        ,'Amber Error' , 'No Error')
    
    df_ai_pandas['Result: Release Held Goods- Multiple Electrical Goods'] = np.where((df_ai_pandas['Result: Release Held Goods- Multiple Electrical Goods']=='Amber Error')&(df_ai_pandas['transcript_x'].str.contains('multiple', case=False, na=False)),'No Error', df_ai_pandas['Result: Release Held Goods- Multiple Electrical Goods'] )
    

#"Was transaction/authorization declined due to credit limit/asked to talk to credit team?"
    df_ai_pandas['Result: Release Held Goods- Large Furniture'] = np.where(
        (df_ai_pandas['Is the call from a store about authorization code/store card referral?'].str.lower() == 'yes' ) & (df_ai_pandas["Was transaction/authorization declined due to credit limit/asked to talk to credit team?"].str.lower() == 'no' ) &

    ((df_ai_pandas['Did the agent ask if the order is for large furniture items only?'].str.lower() == 'no' ) )
        ,'Amber Error' , 'No Error')
    df_ai_pandas['Result: Release Held Goods- Large Furniture'] = np.where((df_ai_pandas['Result: Release Held Goods- Large Furniture']=='Amber Error')&(df_ai_pandas['transcript_x'].str.contains('large furniture', case=False, na=False)),'No Error', df_ai_pandas['Result: Release Held Goods- Large Furniture'] )

    df_ai_pandas['Result: Release Held Goods- Large Furniture'] = np.where((df_ai_pandas['Result: Release Held Goods- Large Furniture']=='Amber Error')&(df_ai_pandas['transcript_x'].str.contains('credit limit', case=False, na=False)),'No Error', df_ai_pandas['Result: Release Held Goods- Large Furniture'] )
    
# "Is the order for large furniture items only?"

    df_ai_pandas['Result: Release Held Goods- other goods ask value exceeds 500 pounds'] = np.where(
        (df_ai_pandas['Is the call from a store about authorization code/store card referral?'].str.lower() == 'yes' ) & (df_ai_pandas['Is the order for large furniture items only?'].str.lower() == 'no' ) & (df_ai_pandas["Was transaction/authorization declined due to credit limit/asked to talk to credit team?"].str.lower() == 'no' ) &
    ((df_ai_pandas['Did the agent ask if the order value exceeds 500 £?'].str.lower() == 'no' ) )
        ,'Amber Error' , 'No Error')

    # df_ai_pandas['Result: Release Held Goods- other goods ask value exceeds 500 pounds'] = np.where(

    # ((df_ai_pandas['Did the agent ask if the order value exceeds 500 £?'].str.lower() == 'no' ) )
    #     ,'Amber Error' , 'No Error')

    df_ai_pandas['Result: Release Held Goods- other goods ask value exceeds 500 pounds'] = np.where((df_ai_pandas['Result: Release Held Goods- other goods ask value exceeds 500 pounds']=='Amber Error')&(df_ai_pandas['transcript_x'].str.contains('500', case=False, na=False)),'No Error', df_ai_pandas['Result: Release Held Goods- other goods ask value exceeds 500 pounds'] )

    df_ai_pandas['Result: Release Held Goods- other goods ask value exceeds 500 pounds'] = np.where((df_ai_pandas['Result: Release Held Goods- other goods ask value exceeds 500 pounds']=='Amber Error')&(df_ai_pandas['transcript_x'].str.contains('credit limit', case=False, na=False)),'No Error', df_ai_pandas['Result: Release Held Goods- other goods ask value exceeds 500 pounds'] )

    

#(df_ai_pandas['Did the agent ask if the order value exceeds 500 £?'].str.lower() == 'no' )
    df_ai_pandas['Result: Release Held Goods- accessible mobile'] = np.where(
        (df_ai_pandas['Is the call from a store about authorization code/store card referral?'].str.lower() == 'yes' ) & (df_ai_pandas['Is the order value exceeds 500?'].str.lower() == 'yes' ) &(df_ai_pandas['Did the agent ask if the order value exceeds 500 £?'].str.lower() == 'yes' ) &

    ((df_ai_pandas['Did the agent ask about the registered mobile number which is accessible?'].str.lower() == 'no' ) )
        ,'Amber Error' , 'No Error')

    df_ai_pandas['Result: Release Held Goods- pin'] = np.where(
        (df_ai_pandas['Is the call from a store about authorization code/store card referral?'].str.lower() == 'yes' ) & (df_ai_pandas['Is the order value exceeds 500?'].str.lower() == 'yes' ) &
        (df_ai_pandas['Did the agent ask if the order value exceeds 500 £?'].str.lower() == 'yes' ) &

    ((df_ai_pandas['Did the agent ask for pin verification if the mobile number is apt?'].str.lower() == 'no' ) )
        ,'Amber Error' , 'No Error')



    # df_ai_pandas['Result: Release Held Goods- phone changed within 10 days'] = np.where(

    # ((df_ai_pandas['Did the agent decline the transaction?'].str.lower() == 'no' ) &(df_ai_pandas['phone_updated_within_10days']))
    #     ,'Amber Error' , 'No Error')

    # df_ai_pandas['Result: Release Held Goods- email changed within 10 days'] = np.where(

    # ((df_ai_pandas['Did the agent decline the transaction?'].str.lower() == 'no' ) &(df_ai_pandas['change_of_email_updated_within10']))
    #     ,'Amber Error' , 'No Error')


    df_ai_pandas['Result: Release Held Goods- fraud check payment done'] = np.where(
        (df_ai_pandas['Is the call from a store about authorization code/store card referral?'].str.lower() == 'yes' ) &

    ((df_ai_pandas['Did the agent decline the transaction?'].str.lower() == 'no' ) &(df_ai_pandas['payment_ever_done'].str.lower()=='no'))
        ,'Amber Error' , 'No Error')
        
    df_ai_pandas['Result: Release Held Goods- Signature at till reciept'] = np.where(
        (df_ai_pandas['Is the call from a store about authorization code/store card referral?'].str.lower() == 'yes' ) &

    ((df_ai_pandas['Did the agent decline the transaction?'].str.lower() == 'no' ) &(df_ai_pandas['Did the agent remind the store consultant to get customer to sign till receipt?'].str.lower()=='no'))
        ,'Compliant with development' , 'No Error')
    
    final_result_cols = [col for col in df_ai_pandas.columns if col.startswith("Result:")]
    df_ai_pandas['iAudit Result: Release Held Goods']= df_ai_pandas[final_result_cols].apply(get_result_priority, axis = 1)
    df_ai_pandas['iAudit Result: Release Held Goods']= np.where(df_ai_pandas['Level 2'].isin(level_2_req_list), df_ai_pandas['iAudit Result: Release Held Goods'], 'N.A.' )
    df_ai_pandas['iAudit Result: Release Held Goods']=np.where(df_ai_pandas["agent_order_max"]== df_ai_pandas["agent_order"],df_ai_pandas['iAudit Result: Release Held Goods'], 'N.A.' )
    df_ai_pandas['iAudit Result: Release Held Goods']=np.where(
                                                ((df_ai_pandas["Output_final"]=="") |(df_ai_pandas["Output_final"].isna()))
                                                
                                                , 'N.A.',df_ai_pandas['iAudit Result: Release Held Goods'] 

        )
    df_ai_pandas['iAudit Result: Release Held Goods'] = np.where((df_ai_pandas['Did the agent decline the transaction?'].str.lower()=='yes') & (df_ai_pandas['iAudit Result: Release Held Goods']!= 'Good Customer Outcome'),'Good Customer Outcome',df_ai_pandas['iAudit Result: Release Held Goods'])


    df_ai_pandas['iAudit Result: Release Held Goods']= np.where(( (df_ai_pandas['Is the conversation in the transcript partial/ended abruptly/incomplete conversation?'].str.lower()== 'yes')| ( df_ai_pandas["Did the agent ask if they have resolved and said bye?"].str.lower()== 'no')), 'Good Customer Outcome',df_ai_pandas['iAudit Result: Release Held Goods'] )
   
    error_list = ["red error", "amber error", "compliant with development"]
    df_ai_pandas['Remarks'] = df_ai_pandas.apply(lambda row: ', '.join([f"{str(row[col]).strip()}: {col.replace('Result: Release Held Goods-', '').strip()}" for col in final_result_cols if str(row[col]).strip().lower() in error_list])
        , axis =1)
    df_ai_pandas['iAudit Score: Release Held Goods']= df_ai_pandas['iAudit Result: Release Held Goods'].map(score_map)	
    df_ai_pandas['iAudit Remarks: Release Held Goods']=	df_ai_pandas['Remarks']
    remarks_dict = {
        "DPA Check": "Customer information not confirmed or value of sale/store number not confirmed",
        "Fraud Check if Virtual Card": "Agent did not ask if the customer is using virtual card through Nextpay app",
        "Fraud Check Virtual Card Screenshot": "Agent did not confirm that customer is using app and not showing just a screenshot",
        "Gift card":"Agent did not confirm the order contains a giftcard",
        "Store Card": "Agent did not confirm if the customer is using store card for retail purchase",
        "Electric Good": "Agent did not confirm if there is an electrical item or branded items over £100 in order",
        "Multiple Electrical Goods": "Agent did not confirm if there were multiple electrical items or branded items over £100 in order",
        "Large Furniture": "Agent did not confirm if the order had large furniture items only",
        "accessible mobile": "Agent did not ask if the mobile phone with registered number accessible",
        "pin": "Agent did not ask for pin to validate mobile number",
        "fraud check payment done": "No payment done for account/not an established account but agent authorised the transaction",
        "Signature at till reciept": "Agent did not ask the customer to sign till receipt",
        "other goods ask value exceeds 500 pounds": "Agent did not confirm if the order value was more than £500"

        }
    df_ai_pandas['iAudit Remarks: Release Held Goods'] = df_ai_pandas['iAudit Remarks: Release Held Goods'].replace(remarks_dict, regex = True)
    df_ai_pandas['iAudit Remarks: Release Held Goods'] = np.where(df_ai_pandas['iAudit Score: Release Held Goods']==10,"",df_ai_pandas['iAudit Remarks: Release Held Goods'] )
    
    df_ai_pandas.to_excel("ReleaseHeldGoodsIntermediateResults.xlsx")

    #confidence scoring
    # def confidence_scoring(row):
    #     result = []
    #     if row['iAudit Remarks: Release Held Goods']=='Customer information not confirmed or value of sale/store number not confirmed':
    #         if row["transcript_x"].str.contains('customer number', case=False, na=False) == False|row["transcript_x"].str.contains('value', case=False, na=False) == False|((row["transcript_x"].str.contains('store number', case=False, na=False) == False) & (row["transcript_x"].str.contains('store id', case=False, na=False) == False)):
    #                result.append("High")
    #         else:
    #             result.append("Low")
    #     if row['iAudit Remarks: Release Held Goods']=='Agent did not confirm the order contains a giftcard':
    #         if row["transcript_x"].str.contains('gift card', case=False, na=False):
    #             result.append("High")
    #         else:
    #             result.append("Low")
    #     if row['iAudit Remarks: Release Held Goods']=='Agent did not confirm if the customer is using store card for retail purchase':
    #         if row["transcript_x"].str.contains('retail', case=False, na=False):
    #             result.append("High")
    #         else:
    #             result.append("Low")
    #     if row['iAudit Remarks: Release Held Goods']=='Agent did not confirm if the customer is using store card for retail purchase':
    #         if row["transcript_x"].str.contains('retail', case=False, na=False):
    #             result.append("High")
    #         else:
    #             result.append("Low")   
    #     if row['iAudit Remarks: Release Held Goods'] == 'Agent did not confirm if there is an electrical item or branded items over £100 in order':
    #         if row["transcript_x"].str.contains('electrical', case=False, na=False):
    #             result.append("High")
    #         else:
    #             result.append("Low")
    #     if row['iAudit Remarks: Release Held Goods']=='Agent did not confirm if the order had large furniture items only':
    #         if row["transcript_x"].str.contains('furniture', case=False, na=False):
    #             result.append("High")
    #         else:
    #             result.append("Low")
    #     if row['iAudit Remarks: Release Held Goods'] =='Agent did not confirm if there were multiple electrical items or branded items over £100 in order':
    #         if row["transcript_x"].str.contains('multiple', case=False, na=False):
    #             result.append("High")
    #         else:
    #             result.append("Low")   

    #     if "Low" in result:
    #         return "Low"
    #     else:
    #         return "High"

    # df_ai_pandas['iAudit Confidence Score: Release Held Goods'] = np.where(df_ai_pandas['iAudit Result: Release Held Goods']=="Good Customer Outcome","High", "N.A." )


    # df_ai_pandas.loc[df_ai_pandas['iAudit Result: Release Held Goods'].isin(['Amber Error', 'Red Error','Compliant with development']), 'iAudit Confidence Score: Release Held Goods'] = df_ai_pandas.loc[df_ai_pandas['iAudit Result: Release Held Goods'].isin(['Amber Error', 'Red Error','Compliant with development'])].apply(confidence_scoring, axis = 1)

    df_ai_pandas.to_excel("ReleaseHeldGoodsIntermediateResults.xlsx")












    df_ai_pandas_filtered = df_ai_pandas[['callid_agentid','callid','AgentId','Queues','iAudit Result: Release Held Goods','iAudit Score: Release Held Goods','iAudit Remarks: Release Held Goods']]

    replace_map_result ={ 'Amber Error': 'Non-Compliant No Poor Outcome',
                      'Red Error': 'Non-Compliant Poor Outcome',
                      'Compliant with development': 'Compliant with Development',
                      'Good Customer Outcome': 'Good Customer Outcome',
                      'N.A.': 'N.A.'
                      


        }
    df_ai_pandas_filtered['iAudit Result: Release Held Goods'] = df_ai_pandas_filtered['iAudit Result: Release Held Goods'].replace(replace_map_result).replace(['', None, np.nan], 'N.A.')
    df_ai_pandas['iAudit Score: Release Held Goods']= np.where(
            df_ai_pandas_filtered['iAudit Result: Release Held Goods'].str.contains('N.A.'),10, df_ai_pandas_filtered['iAudit Score: Release Held Goods']
            )
    
    df_ai_pandas_filtered = df_ai_pandas_filtered.drop_duplicates(subset=['callid_agentid'], keep='first'
    )

    print("Final Shape",df_ai_pandas_filtered.shape)






    df_ai_pandas_filtered.to_excel("ReleaseHeldGoodsFinalResults0406.xlsx")





















    
















