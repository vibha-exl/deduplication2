!pip install sentence-transformers==5.1.0
!pip install json-repair -q
!pip install openpyxl -q
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
from TransfersCorrectlyPromptsNew import *
from Transfer_Matrix_Reasons import *
from pyspark.sql.functions import format_string
from pyspark.sql.functions import array, lit, when, size, concat_ws
from pyspark.sql import functions as F
from sentence_transformers import SentenceTransformer, util
import yaml
from pyspark.sql import SparkSession
spark = SparkSession.builder.getOrCreate()
from json_repair import repair_json

warnings.filterwarnings("ignore")

log = get_logger()

def safe_repair_and_load(s):
    try:
        repaired = repair_json(s)
        return json.loads(s)
    except Exception:
        return None

def get_best_match(output_embedding,target_embeddings, target_sentences):
    similarities =  util.pytorch_cos_sim(output_embedding, target_embeddings)
    best_match_idx = similarities.argmax().item()
    return(target_sentences[best_match_idx])
  
def get_best_match_score(output_embedding,target_embeddings):
    similarities =  util.pytorch_cos_sim(output_embedding, target_embeddings)
    scores = similarities[0]
    best_score = scores.max().item()
    return(best_score)  
  

def transfers_correctly_main_function(transcripts):
    transcripts =transcripts.rename(columns={'AgentID':'AgentId'})
    # transcripts =transcripts.rename(columns={'Callid':'callid'})
    #print(transcripts.columns)
    meta_reasons = pd.DataFrame(reason_matrix_dict)
    meta_reasons['r1'] = meta_reasons['REASON OF TRANSFER'].fillna("").str.strip()
    meta_reasons['r2'] = meta_reasons['Descriptive Reason of Transfer'].fillna("").str.strip()

    meta_reasons['FinalReason'] = np.where((meta_reasons['r1'].str.lower()== meta_reasons['r2'].str.lower()), meta_reasons['r1'], meta_reasons['r1'] + " - " + meta_reasons['r2'])
    

    num_unique_callid_agentid = transcripts[['callid', 'AgentId']].drop_duplicates().shape[0]
    # print("num_unique_callid_agentid transcript", num_unique_callid_agentid)
    # print("transcripts", transcripts.shape)
    log.info(f"Unique combination callid-agentid {num_unique_callid_agentid}")
    log.info(f"Initial transcripts shape {transcripts.shape}")
    

    

    total_calls = len(transcripts['callid'].unique().tolist())
    filtered_calls = transcripts['callid'].unique().tolist()[:total_calls]
    #filtered_calls = transcripts['callid'].unique().tolist()[:100]
    df_transcripts = transcripts[transcripts['callid'].isin(filtered_calls)].reset_index(drop=True)
    # print("df_transcripts", df_transcripts.shape)

    audit_callids = df_transcripts['callid'].unique().tolist()

    # print("audit_callids", len(audit_callids))
    log.info(f"Unique callids {len(audit_callids)}")

  

    data = df_transcripts[['callid', 'AgentId', 'channel', 'transcript', 'createdate']]


    data['callid_agentid']= data['callid'] + "|" + data['AgentId']
    # print("data",data.columns)
    grouped_transcript_all = data.groupby('callid_agentid').apply(lambda x: "\n".join(f"{row['channel']}: {row['transcript']}" for _, row in x.iterrows())).reset_index(name = 'transcript')
    data_intermediate = data.groupby('callid_agentid', as_index = False).first()
    data_req = pd.merge(grouped_transcript_all, data_intermediate, on='callid_agentid', how='left')
    data_req['createdate'] = pd.to_datetime(data_req['createdate'], format = 'mixed', errors='coerce')
    data_req = data_req.sort_values(by=["callid", 'createdate'], ascending =[True, True])
    data_req['createdate'] = data_req['createdate'].astype(str)
    data_req['agent_order'] = data_req.groupby("callid").cumcount()+1
    data_req['agent_order_max'] = data_req.groupby("callid")["agent_order"].transform("max")

    callids_with_multiple_agents = data_req.loc [data_req['agent_order_max'] > 1, 'callid'].unique().tolist()

    # print("data_req" ,data_req.shape )


    # print( "multiple_agents_callids",len(callids_with_multiple_agents ))
    log.info(f"Callids with multiple agents acc to transcripts {len(callids_with_multiple_agents )}")
    

    # data_req.to_excel("transfers_intermediatedata.xlsx")


    spark_transfers =  spark.sql(f"""
            SELECT * FROM `contactcentre_prod`.`iaudit`.`vw_transfers` 
            WHERE conversationid IN {tuple(callids_with_multiple_agents)}      
        """)

    df_transfers = spark_transfers.toPandas() 

    #df_transfers.to_excel("transfers_table.xlsx")

    df_transfers['callid_agentid']= df_transfers['conversationid'] + "|" + df_transfers['transferring_agentid']
    df_transfers = df_transfers.groupby('callid_agentid', as_index = False).first()

    data_req = pd.merge(data_req, df_transfers, on='callid_agentid', how='left')
    data_req['dest_dept'] = data_req['dest_dept'].replace(['', ' '], pd.NA).fillna('N.A.')
    data_req['eligibility'] = np.where(

      (data_req['dest_dept'] == 'N.A.') | (data_req["agent_order_max"]==data_req["agent_order"]), 0,1
    )

    #account_type payin3

    # spark_logger =  spark.sql(f"""
    #             SELECT * FROM contactcentre_prod.staging.zen_live9 
    #             WHERE conversation_id IN {tuple(callids_with_multiple_agents)}      
    #         """)

    # df_logger = spark_logger.toPandas()

    # d1 = {'id': 'ticket_id',
    #         'Classification': '',
    #         'call_date': '',
    #         'channel': 'Channel_logger',
    #         'CallResolution': '',
    #         'CallType': '',
    #         'DepartmentName': '',
    #         'WorkGroupName': 'Queues',
    #         'IsAuditable': '',
    #         'TreeVersion': '',
    #         'isexpressionofdissatisfaction': '',
    #         'account_number': '',
    #         'account_type': '',
    #         'ICW_reference': '',
    #         'assignee': '',
    #         'conversation_id': 'CallId',
    #         'CountryName': '',
    #         'Supplier': '',
    #         'Adjustment_Value': '',
    #         'Compensation_value': '',
    #         'Vulnerable_customer_caseid': 'Vulnerable Customer Case ID',
    #         'Vulnerable_Customer': 'Vulnerable Customer',
    #         'RootResponseName': 'Level 1',
    #         'ResponseName': 'Level 2',
    #         'ReasonL1': '',
    #         'ReasonL2': '',
    #         'RootResponseName1': '',
    #         'ResponseName1': '',
    #         'Vulnerable_Customer1_a': 'Vuln Support Offered',
    #         'tTalk': '',
    #         'tHold': '',
    #         'tAcw': '',
    #         'created_at': '',
    #         'Sub_Department': ''}

    # rename_cols = {k:v for k,v in d1.items() if v!=''}
    # df_logger = df_logger.rename(columns=rename_cols)
    # print("Logger unique callids", len(df_logger['CallId'].unique().tolist()))
    # customer_account_number_list = df_logger['account_number'].unique().tolist()
    # customer_account_number_list = [x for x in customer_account_number_list if str(x) != 'nan']
    # print("unique_account_number",len(customer_account_number_list))
    # customer_account_number_list = [ x for x in customer_account_number_list if x is not None]
    # print("Not none unique_account_number",len(customer_account_number_list))
    # df_logger = df_logger.drop_duplicates(subset = 'CallId', keep='first')

    # spark_account_details =  spark.sql(f"""
    #     SELECT `account_number`, `AccountType`
    #     FROM `contactcentre_prod`.`iaudit`.`account_details`
    #     WHERE `account_number` IN {tuple(customer_account_number_list)}  
    #     """)
    # df_account_details = spark_account_details.toPandas()
    # print(df_account_details.shape)
    # df_account_details= df_account_details.drop_duplicates(subset = 'account_number', keep='first')
    # print("Account details final",df_account_details.shape)

    # account_number_account_details_list = df_account_details['account_number'].unique().tolist()
    # print("Account Details unique account number", len(account_number_account_details_list))
    # # df_account_details = df_account_details.rename(columns={'AccountNumber': 'account_number'})
    # df_logger = df_logger[['CallId', 'account_number', 'Level 2','Queues']]
    # df_logger = df_logger.merge(df_account_details, on='account_number', how='left')
    # data_req = data_req.merge(df_logger,  left_on='callid', right_on='CallId', how='left')
    # data_req['account_number'] = data_req['account_number'].astype(str)
    # data_req['AccountType'].replace('',np.nan, inplace = True)
    # data_req['AccountType'].fillna('N.A.', inplace = True)
        


    #Warm transfer

    data_req['warm_transfer_check'] = np.where(
      ((data_req['eligibility']==1) & (data_req['dest_dept'].isin(['CCAIT', 'CET', 'CRMT'])) & (data_req['transfer_mode']!= 'Warm transfer')), 'No Evidence', 'Full Evidence'

    )
    data_req['warm_transfer_check_comment'] = np.where(data_req['warm_transfer_check'] == 'No Evidence', 'transfer_mode in DB: '+data_req['transfer_mode']+ ' But dest_dept in DB: '+data_req['dest_dept'], ""
                                                       )
    log.info(f"LLM hits to start..")

    # data_req.to_excel("transfers_intermediatedata.xlsx")
    df_prompts = (
                    spark.createDataFrame(data_req)
                    .withColumn( "user_prompt", F.when((F.col("eligibility")==1) , F.lit(user_prompt_transfers_correctly))
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
    
    # df_ai = (
    #             df_prompts.withColumn(
    #                 "ai_response", F.when(
    #                     (F.col("eligibility")==1) &
    #                 (F.col("user_prompt") != ""), F.expr(
    #                     """ai_query('databricks-meta-llama-3-1-8b-instruct',request => final_prompt)"""
    #                         )
                    
    #             )
    #         )
    #         )
    # display(df_ai.toPandas())
    #contact_centre_internal_batch
    df_ai = (
                df_prompts.withColumn(
                    "ai_response", F.when(
                        (F.col("eligibility")==1) &
                    (F.col("user_prompt") != ""), F.expr(
                        """ai_query('contact_centre_internal_batch',request => final_prompt)"""
                            )
                    
                )
            )
            )
    

    df_ai_pandas = df_ai.toPandas()
    #df_ai_pandas.to_excel("transfers_intermediatedata.xlsx")

    # print("LLM hits done",df_ai_pandas.shape )
    log.info(f"LLM hits done {df_ai_pandas.shape}")

    if 'ai_response' not in df_ai_pandas.columns:
            df_ai_pandas['ai_response'] = ""
    df_ai_pandas['ai_response1'] = (df_ai_pandas['ai_response']
                                    .str.replace("```json", "", regex = False)
                                    .str.replace("```", "",regex = False)
                                    .str.strip()
                                    )
    df_ai_pandas['Output_final'] = df_ai_pandas['ai_response1'].apply(safe_repair_and_load)
    df_ai_pandas = pd.concat([df_ai_pandas,df_ai_pandas['Output_final'].apply(pd.Series)], axis =1) 

    ai_response_column_list = [
        "Summarized_reason_transfer",
        "Are there multiple intents/reasons of caller in the transcript/which need different transfers?",
        "Did the agent directly transferred the call after listening to caller query/after customer identification?",
        "Did the caller ask the agent to transfer the call to a particular department/queue/person?",
        "Does the transcript seem abruptly incomplete before transfer?",
        "Was there any injury or harm to the caller due to a product?",
        "Is the caller calling about goods not recieved?",
        "Did the transcript talk about an international order?",
        "Is there conversation about branded items?",
        "Is the caller facing financial difficulties/under stress?",
        "CONTEXT"
      
                 ]
    
    for col in ai_response_column_list:
            if col not in df_ai_pandas.columns:
                df_ai_pandas[col] = "N.A."

    df_ai_pandas[ai_response_column_list] = df_ai_pandas[ai_response_column_list].replace("", np.nan).fillna("N.A.")          

    
    model = SentenceTransformer('all-MiniLM-L6-v2')
    df_ai_pandas['LLM_cust_query_output'] = df_ai_pandas['Summarized_reason_transfer'].fillna("N.A.").astype(str)
    output_embeddings = model.encode(df_ai_pandas['LLM_cust_query_output'].tolist(), convert_to_tensors = True)
    target_sentences = meta_reasons["FinalReason"].tolist()
    target_embeddings = model.encode(target_sentences, convert_to_tensors = True)
    df_ai_pandas['matched_reason'] = [get_best_match(embedding,target_embeddings,target_sentences) for embedding in output_embeddings]
    df_ai_pandas['similarity_score'] =[get_best_match_score(embedding,target_embeddings) for embedding in output_embeddings]
    df_ai_pandas['matched_reason'] = np.where(df_ai_pandas['LLM_cust_query_output'] =='N.A.', 'N.A.',df_ai_pandas['matched_reason'] )
    df_ai_pandas['confidence'] = np.where(df_ai_pandas['similarity_score'] >= 0.7, 'High', 'Low')

    df_ai_pandas =  pd.merge(df_ai_pandas, meta_reasons, left_on='matched_reason', right_on ='FinalReason' ,how='left')

    df_ai_pandas.to_excel("transfers_intermediatedata.xlsx")

    replace_map_result ={ 'Enquiries EXL': 'Enquiries',
                          'CET': 'CRMT',
                          'Enquiries UK': 'Enquiries'

        }
     
    benefit_doubt_depts = ['iHelp', 'Courier', 'CCx Systems', 'Business Systems', 'Non-Operational', 'Admin', 'Non-Operation'] 

    df_ai_pandas['dest_dept_transformed'] = df_ai_pandas['dest_dept'].replace(replace_map_result)

    df_ai_pandas['transfer_correctly'] = np.where(
      df_ai_pandas['dest_dept_transformed'] == df_ai_pandas['Dept(transfermatrix)'], "Full Evidence", "No Evidence"


    )
    
    df_ai_pandas['transfer_correctly'] = np.where(
      (df_ai_pandas['Dept(transfermatrix)'] =='')|(pd.isna(df_ai_pandas['Dept(transfermatrix)']))  , "Full Evidence", df_ai_pandas['transfer_correctly']

    )
    df_ai_pandas['transfer_correctly'] = np.where(
      df_ai_pandas['dest_dept_transformed'].isin(benefit_doubt_depts) , "Full Evidence", df_ai_pandas['transfer_correctly']

    )
    df_ai_pandas['transfer_correctly'] = np.where(
      df_ai_pandas['LLM_cust_query_output'] =='N.A.' , "N.A.", df_ai_pandas['transfer_correctly']

    )

    

    comment_str = ("Transferred to "+df_ai_pandas['dest_dept']+" but should be transferred to "+df_ai_pandas['Dept(transfermatrix)']+"; \n Matched Reason(from matrix): "+ df_ai_pandas['REASON OF TRANSFER']+"; \n Summarized Reason extracted from LLM: "+df_ai_pandas['Summarized_reason_transfer']+"\n Based on Context: \n "+df_ai_pandas['CONTEXT']+ "; \n Was the call transferred directly by agent:"+ df_ai_pandas["Did the agent directly transferred the call after listening to caller query/after customer identification?"]  )

    df_ai_pandas['transfer_correctly_comment'] = np.where(
      (df_ai_pandas['transfer_correctly'] == "No Evidence"), comment_str, ""
    )

    df_ai_pandas['iAudit Result Critical Question: Transfers Correctly'] =df_ai_pandas['transfer_correctly']
    df_ai_pandas['iAudit Result Critical Question: Transfers Correctly'] = np.where( df_ai_pandas['warm_transfer_check'] == 'No Evidence', 'No Evidence', df_ai_pandas['iAudit Result Critical Question: Transfers Correctly']

    )
    df_ai_pandas['iAudit Comment Critical Question: Transfers Correctly'] = np.where(df_ai_pandas['warm_transfer_check'] == 'No Evidence' , df_ai_pandas['warm_transfer_check_comment'], df_ai_pandas['transfer_correctly_comment']

    )

    df_ai_pandas['iAudit Result Critical Question: Transfers Correctly'] = np.where(df_ai_pandas['eligibility']==0, 'N.A.',  df_ai_pandas['iAudit Result Critical Question: Transfers Correctly'])

    df_ai_pandas['iAudit Result Critical Question: Transfers Correctly'] = np.where(((data_req["agent_order_max"]==data_req["agent_order"]) &(data_req['dest_dept'] != 'N.A.')), 'Full Evidence',  df_ai_pandas['iAudit Result Critical Question: Transfers Correctly'])

    df_ai_pandas['iAudit Result Critical Question: Transfers Correctly'] =  np.where((df_ai_pandas['similarity_score'] <= 0.6) & (df_ai_pandas['iAudit Result Critical Question: Transfers Correctly'] == 'No Evidence') , 'Full Evidence',df_ai_pandas['iAudit Result Critical Question: Transfers Correctly'] )
#dest_dept
    df_ai_pandas['iAudit Result Critical Question: Transfers Correctly'] = np.where(((df_ai_pandas['iAudit Result Critical Question: Transfers Correctly'] =='No Evidence') & (df_ai_pandas['transcript_x'].str.contains('branded', case=False, na=False)) & ((df_ai_pandas['dest_dept'].str.contains('Home', case= False, na = False))|(df_ai_pandas['dest_dept'].str.contains('TP', case= False, na = False)) )), "Full Evidence", df_ai_pandas['iAudit Result Critical Question: Transfers Correctly'])

    df_ai_pandas['iAudit Result Critical Question: Transfers Correctly'] = np.where( ((df_ai_pandas['iAudit Result Critical Question: Transfers Correctly'] =='No Evidence')& ((df_ai_pandas['dest_dept'].str.contains('Home', case= False, na = False))|(df_ai_pandas['dest_dept'].str.contains('TP', case= False, na = False)) ) & (df_ai_pandas['Is there conversation about branded items?'].str.lower()=='yes') ), 'Full Evidence', df_ai_pandas['iAudit Result Critical Question: Transfers Correctly'])


  

    df_ai_pandas['iAudit Result Critical Question: Transfers Correctly'] = np.where(((df_ai_pandas['iAudit Result Critical Question: Transfers Correctly'] =='No Evidence') & (df_ai_pandas['transcript_x'].str.contains('flower', case=False, na=False)) & (df_ai_pandas['dest_dept'].str.contains('Enquiries', case= False, na = False))), "Full Evidence", df_ai_pandas['iAudit Result Critical Question: Transfers Correctly'])

    df_ai_pandas['iAudit Result Critical Question: Transfers Correctly'] = np.where(((df_ai_pandas['iAudit Result Critical Question: Transfers Correctly'] =='No Evidence') & (df_ai_pandas['transcript_x'].str.contains('complaint|customer care', case=False, na=False)) & (df_ai_pandas['dest_dept'].str.contains('CET', case= True, na = False))), "Full Evidence", df_ai_pandas['iAudit Result Critical Question: Transfers Correctly'])

    #"Was there any injury or harm to the caller due to a product?"
    df_ai_pandas['iAudit Result Critical Question: Transfers Correctly'] = np.where(
        (df_ai_pandas["Was there any injury or harm to the caller due to a product?"].str.lower()=='yes') & (df_ai_pandas['dest_dept'].str.contains('CET', case= True, na = False)), 'Full Evidence', df_ai_pandas['iAudit Result Critical Question: Transfers Correctly']

    )
    #"Is the caller calling about goods not recieved?"
    df_ai_pandas['iAudit Result Critical Question: Transfers Correctly'] = np.where(
        (df_ai_pandas["Is the caller calling about goods not recieved?"].str.lower()=='yes') & (df_ai_pandas['dest_dept'].str.contains('Enquiries', case= False, na = False)), 'Full Evidence', df_ai_pandas['iAudit Result Critical Question: Transfers Correctly']

    )


    home_furniture_phrases_list = ['furniture', 'curtains', 'home team', 'mattress','table','tables', 'chair', 'bed', 'beds', 'drawer','sofa', 'mirror', 'Personalised gift', 'personalized gift', 'DHL Delivery', "sofa set", "couch",
                    "dining table",
                    "mattresses",
                    "dressing tables",
                    "wardrobes",
                    "chest of drawers",
                    "coffee tables",
                    "dining bench sets",
                    "dining chairs",
                    "shelves",
                    "sideboards",
                    "side tables",
                    "stools",
                    "stool",
                    "ottomans",
                    "ottoman",
                    "TV units",
                    "TV unit",
                    "wardrobes"
    ]
    pattern_home = "|".join(home_furniture_phrases_list)

    df_ai_pandas['iAudit Result Critical Question: Transfers Correctly'] = np.where(
        (df_ai_pandas['iAudit Result Critical Question: Transfers Correctly'] == 'No Evidence') &
        (df_ai_pandas['transcript_x'].str.contains(pattern_home, case=False, na=False)) &
        (df_ai_pandas['dest_dept'].str.contains('Home', case=False, na=False)),
        "Full Evidence",
        df_ai_pandas['iAudit Result Critical Question: Transfers Correctly']
    )
    enquiry_phrases_list = ['direct debit', 'credit limit','increasing credit limit', 'place an order','decreasing credit limit','close account', 'closing an account', 'gift card', 'ParcelShop', 'faulty', 'pay 3', 'Pay in 3', 'three-step account', 'pay in three', 'paid 3 account','tracking an order', "bathroom accesories", "light", "blind", "blinds", "swatches", "small furniture", "swatch", 
        "towels", "bathmats",
        "bathroom furniture", "bathroom",
        "lightning products",
        "cushions", "cushion",
        "rugs", "rug",
        "runners", "mats",
        "bedsheets", "bedsheet",
        "pillows", "pillow",
        "duvets", "duvet",
        "curtains", "curtain",
        "floor Lamps",
        "ceiling Lamp",
        "mirror", "lamp", "lamps",
        "vase", "vases"
    ]
    pattern_enquiry = "|".join(enquiry_phrases_list)
    df_ai_pandas['iAudit Result Critical Question: Transfers Correctly'] = np.where(
        (df_ai_pandas['iAudit Result Critical Question: Transfers Correctly'] == 'No Evidence') &
        (df_ai_pandas['transcript_x'].str.contains(pattern_enquiry, case=False, na=False)) &
        (df_ai_pandas['dest_dept'].str.contains('Enquiries', case=False, na=False)),
        "Full Evidence",
        df_ai_pandas['iAudit Result Critical Question: Transfers Correctly']
    )

    ccait_phrases_list = ['fraud','account blocked', 'credit limit', 'placed on hold', 'authorisation', 'authorization', 'code', 'store card referral', 'payment plan']
    pattern_ccait = "|".join(ccait_phrases_list)

    df_ai_pandas['iAudit Result Critical Question: Transfers Correctly'] = np.where(
        (df_ai_pandas['iAudit Result Critical Question: Transfers Correctly'] == 'No Evidence') &
        (df_ai_pandas['transcript_x'].str.contains(pattern_ccait, case=False, na=False)) &
        (df_ai_pandas['dest_dept'].str.contains('CCAIT', case=False, na=False)),
        "Full Evidence",
        df_ai_pandas['iAudit Result Critical Question: Transfers Correctly']
    )

#"Did the transcript talk about an international order?"
    df_ai_pandas['iAudit Result Critical Question: Transfers Correctly'] = np.where((df_ai_pandas['Did the transcript talk about an international order?'].str.lower() == 'no') &(df_ai_pandas['iAudit Result Critical Question: Transfers Correctly'] == 'No Evidence') &(df_ai_pandas['matched_reason'] == 'Refund Not Received on the International Account'), 'Full Evidence', df_ai_pandas['iAudit Result Critical Question: Transfers Correctly'] )


    df_ai_pandas['iAudit Result Critical Question: Transfers Correctly'] = np.where((df_ai_pandas['Did the transcript talk about an international order?'].str.lower() == 'yes') &(df_ai_pandas['iAudit Result Critical Question: Transfers Correctly'] == 'No Evidence') &(df_ai_pandas['dest_dept'].str.contains('International', case=False, na=False)), 'Full Evidence', df_ai_pandas['iAudit Result Critical Question: Transfers Correctly'] )



#"Is the caller facing financial difficulties/under stress?"
    df_ai_pandas['iAudit Result Critical Question: Transfers Correctly'] = np.where((df_ai_pandas['Is the caller facing financial difficulties/under stress?'].str.lower() == 'yes') &(df_ai_pandas['iAudit Result Critical Question: Transfers Correctly'] == 'No Evidence') &(df_ai_pandas['dest_dept'].str.contains('CCAIT', case=False, na=False)), 'Full Evidence', df_ai_pandas['iAudit Result Critical Question: Transfers Correctly'] )
    

    df_ai_pandas['agent_utterances_count'] = df_ai_pandas['transcript_x'].str.count('AGENT:')
    df_ai_pandas['customer_utterances_count'] = df_ai_pandas['transcript_x'].str.count('CALLER:')
    df_ai_pandas['total_utterances_count'] = df_ai_pandas['agent_utterances_count'] + df_ai_pandas['customer_utterances_count']


    df_ai_pandas['iAudit Result Critical Question: Transfers Correctly'] = np.where((df_ai_pandas['iAudit Result Critical Question: Transfers Correctly'] == 'No Evidence') &((df_ai_pandas['Does the transcript seem abruptly incomplete before transfer?'].str.lower() == 'yes')|(df_ai_pandas['total_utterances_count']<5)) , 'Full Evidence', df_ai_pandas['iAudit Result Critical Question: Transfers Correctly'])


    df_ai_pandas['iAudit Result Critical Question: Transfers Correctly'] = np.where((df_ai_pandas['iAudit Result Critical Question: Transfers Correctly'] == 'No Evidence') &(df_ai_pandas['Did the caller ask the agent to transfer the call to a particular department/queue/person?'].str.lower() == 'yes') , 'Full Evidence', df_ai_pandas['iAudit Result Critical Question: Transfers Correctly'])

    # #payin3
    # df_ai_pandas['iAudit Result Critical Question: Transfers Correctly'] = np.where(((df_ai_pandas['iAudit Result Critical Question: Transfers Correctly'] == 'No Evidence') &(df_ai_pandas['AccountType'] == 'PayIn3') & (df_ai_pandas['dest_dept'].str.contains('Enquiries', case=False, na=False))) , 'Full Evidence', df_ai_pandas['iAudit Result Critical Question: Transfers Correctly'])

    df_ai_pandas['iAudit Confidence Critical Question: Transfers Correctly'] = np.where(df_ai_pandas['iAudit Result Critical Question: Transfers Correctly']== 'No Evidence', df_ai_pandas['confidence'], 'High')

    df_ai_pandas['iAudit Confidence Critical Question: Transfers Correctly'] = np.where(((df_ai_pandas['iAudit Result Critical Question: Transfers Correctly']== 'No Evidence') | (df_ai_pandas['iAudit Result Critical Question: Transfers Correctly']== 'Full Evidence')) & (df_ai_pandas['Did the agent directly transferred the call after listening to caller query/after customer identification?'].str.lower() == 'yes'), "High", df_ai_pandas['iAudit Confidence Critical Question: Transfers Correctly'])

    # df_ai_pandas['iAudit Confidence Critical Question: Transfers Correctly'] = np.where(df_ai_pandas['iAudit Result Critical Question: Transfers Correctly']== 'Full Evidence',"High" , df_ai_pandas['iAudit Confidence Critical Question: Transfers Correctly'])

   
    # df_ai_pandas['iAudit Result Critical Question: Transfers Correctly'] = np.where( df_ai_pandas['warm_transfer_check'] == 'No Evidence', 'No Evidence', df_ai_pandas['iAudit Result Critical Question: Transfers Correctly']

    # )
    # df_ai_pandas['iAudit Comment Critical Question: Transfers Correctly'] = np.where(df_ai_pandas['warm_transfer_check'] == 'No Evidence' , df_ai_pandas['warm_transfer_check_comment'], df_ai_pandas['iAudit Comment Critical Question: Transfers Correctly']

    # )

    df_ai_pandas['iAudit Comment Critical Question: Transfers Correctly'] = np.where(df_ai_pandas['iAudit Result Critical Question: Transfers Correctly']== 'Full Evidence', '',  df_ai_pandas['iAudit Comment Critical Question: Transfers Correctly'])

    df_ai_pandas['iAudit Result Critical Question: Transfers Correctly'] = np.where(df_ai_pandas['eligibility']==0, 'N.A.',  df_ai_pandas['iAudit Result Critical Question: Transfers Correctly'])

    df_ai_pandas['iAudit Result Critical Question: Transfers Correctly'] = np.where(((data_req["agent_order_max"]==data_req["agent_order"]) &(data_req['dest_dept'] != 'N.A.')), 'Full Evidence',  df_ai_pandas['iAudit Result Critical Question: Transfers Correctly'])


    df_ai_pandas['iAudit Comment Critical Question: Transfers Correctly'] = np.where(df_ai_pandas['eligibility']==0, '',  df_ai_pandas['iAudit Comment Critical Question: Transfers Correctly'])
    df_ai_pandas['iAudit Score Critical Question: Transfers Correctly'] = np.where(df_ai_pandas['iAudit Result Critical Question: Transfers Correctly']== 'No Evidence', 0, 10

    )
    df_ai_pandas.to_excel("transfers_intermediatedata.xlsx")

    df_ai_pandas_final = df_ai_pandas[['callid','callid_agentid','AgentId','iAudit Score Critical Question: Transfers Correctly','iAudit Result Critical Question: Transfers Correctly','iAudit Comment Critical Question: Transfers Correctly', 'iAudit Confidence Critical Question: Transfers Correctly']]

    log.info(f"Final shape: {df_ai_pandas_final.shape}")

    

    return df_ai_pandas_final

if __name__=="__main__":

    formatted_date ='2026-02-21'
   
    # transcripts = pd.read_excel(r'/Workspace/Users/ananthagiri_abhiram@next.co.uk/iAudit_Trisha/UATpool2CallsRedcatedTranscripts.xlsx')
    # transcripts = pd.read_excel(r'/Workspace/Users/ananthagiri_abhiram@next.co.uk/iAudit_Trisha/auditedcallsjan2026.xlsx')
    # transcripts = pd.read_excel(r'/Workspace/Users/ananthagiri_abhiram@next.co.uk/iAudit_Trisha/callstransferserror.xlsx')


    # transcripts = transcripts[transcripts['callid'].isin(["073c548a-b1c1-4a8d-a75f-4a8c07e742a2",
    #   "1046d760-6886-4f81-b65f-6c5d8cb8cdf9",
    #   "4b24a378-743b-4750-9938-a21c361bab9a",
    #   "7267787f-d905-4386-86a4-8291e1dfdb25",
    #   "9402c434-e2e3-4220-8412-be32bfe57baa",
    #   "97beac06-0802-4521-965e-9c574aeac385",
    #   "9d3f63ea-1e98-4e6b-b9db-3ce4ff7bf9e0",
    #   "b0ff3684-714b-4deb-9d6d-f578c1a84264",
    #   "b493c579-fc19-4640-81bf-3dcfa0227204"
    #   ])]
   
    raw_table_query = f"""SELECT * FROM contactcentre_prod.transcripts.transcripts_raw WHERE DATE(createdate) = '{formatted_date}'"""
   
    full_raw_table = spark.sql(raw_table_query)
    transcripts = full_raw_table.toPandas()

    df_ai_pandas_final = transfers_correctly_main_function(transcripts)
    df_ai_pandas_final.to_excel("TransfersCorrectlyFinalResults21feb.xlsx")




    











    

















