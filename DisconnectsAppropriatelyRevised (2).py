# !pip install PyYAML
# !pip install openpyxl
import warnings
import os
import json
#from json_repair import repair_json
import re
from datetime import datetime
import numpy as np
import pandas as pd
from pyspark.errors import PySparkException
from pyspark.sql.types import StructType, StructField, BooleanType, StringType
from pyspark.sql.functions import from_json, col, lit, udf, when, expr
from iaudit_logger import get_logger
from DisconnectsAppropriatelyPrompts import *
from Transfer_Matrix_Reasons import *
from pyspark.sql.functions import format_string
from pyspark.sql.functions import array, lit, when, size, concat_ws
from pyspark.sql import functions as F
# from sentence_transformers import SentenceTransformer, util
import yaml
from pyspark.sql import SparkSession
spark = SparkSession.builder.getOrCreate()
warnings.filterwarnings("ignore")

log = get_logger()

def last_n_words(text, n=100):
    words = text.split()
    return " ".join(words[-n:])

def disconnects_appropriately_main_function(transcripts):
    transcripts =transcripts.rename(columns={'AgentID':'AgentId'})
    # transcripts =transcripts.rename(columns={'Callid':'callid'})
    #print(transcripts.columns)
    meta_reasons = pd.DataFrame(reason_matrix_dict)

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
    print("data",data.columns)
    grouped_transcript_all = data.groupby('callid_agentid').apply(lambda x: "\n".join(f"{row['channel']}: {row['transcript']}" for _, row in x.iterrows())).reset_index(name = 'transcript')
    data_intermediate = data.groupby('callid_agentid', as_index = False).first()
    data_req = pd.merge(grouped_transcript_all, data_intermediate, on='callid_agentid', how='left')
    data_req['createdate'] = pd.to_datetime(data_req['createdate'], format = 'mixed', errors='coerce')
    data_req = data_req.sort_values(by=["callid", 'createdate'], ascending =[True, True])
    data_req['createdate'] = data_req['createdate'].astype(str)
    data_req['agent_order'] = data_req.groupby("callid").cumcount()+1
    data_req['agent_order_max'] = data_req.groupby("callid")["agent_order"].transform("max")

    spark_transfers =  spark.sql(f"""
            SELECT * FROM `contactcentre_prod`.`iaudit`.`vw_transfers` 
            WHERE conversationid IN {tuple(audit_callids)}      
        """)

    df_transfers = spark_transfers.toPandas()

    transfer_calls = df_transfers['conversationid'].unique().tolist()

    data_req['transferred'] = data_req['callid'].isin(transfer_calls).astype(int)

    # data_req.to_excel("intermediate_disconnects_appropriately.xlsx")



    spark_disconnect_type =  spark.sql(f"""
     SELECT * FROM `contactcentre_prod`.`iaudit`.`vw_disconnect_direction_acw` 
            WHERE conversationId IN {tuple(audit_callids)}      
        """)

    df_disconnect_type = spark_disconnect_type.toPandas()
    # print("df_disconnect_type before", df_disconnect_type.shape)
    log.info(f"df_disconnect_type before {df_disconnect_type.shape}")
    df_disconnect_type['userId'] = df_disconnect_type['userId'].replace('', np.nan)
    df_disconnect_type = df_disconnect_type.dropna(subset =['userId'])
    df_disconnect_type['callid_agentid']= df_disconnect_type['conversationId'] + "|" + df_disconnect_type['userId']
    df_disconnect_type['DisconnectType'] = df_disconnect_type['DisconnectType'].replace('', np.nan)
    df_disconnect_type['has_DisconnectType'] = df_disconnect_type['DisconnectType'].notna()
    df_disconnect_type = ( df_disconnect_type.sort_values('has_DisconnectType', ascending =False).groupby('callid_agentid', as_index = False).first().drop(columns = 'has_DisconnectType')
    )
    # print("df_disconnect_type after", df_disconnect_type.shape)
    log.info(f"df_disconnect_type after {df_disconnect_type.shape}")
    df_disconnect_type.to_excel("Disconnect_type_table.xlsx")

    data_req = pd.merge(data_req, df_disconnect_type, on='callid_agentid', how='left')
    data_req['DisconnectType'] = data_req['DisconnectType'].replace(['', ' '], pd.NA).fillna('N.A.')
    data_req['eligibility'] = np.where(

      ((data_req['DisconnectType'] != 'N.A.') & (data_req["agent_order_max"]==data_req["agent_order"] ) & (data_req['DisconnectType'].isin(['client']))), 1,0
    )
    data_req['end_transcript'] = data_req['transcript_x'].apply(lambda x: last_n_words(x, 100))

    log.info(f"LLM hits to start..")

    df_prompts = (
                    spark.createDataFrame(data_req)
                    .withColumn( "user_prompt", F.when((F.col("eligibility")==1) , F.lit(user_prompt_disconnects_appropriately))
                                .otherwise(F.lit(""))
                                
                                )
                    .withColumn("final_prompt", F.when( (F.col("eligibility")==1),
                                                F.concat_ws( "\n\n",
                                                            F.lit("SYSTEM PROMPT:"),
                                                            F.lit(system_prompt),
                                                            F.lit("USER PROMPT:"),
                                                            F.col("user_prompt"),
                                                            F.lit("END of TRANSCRIPT:"),
                                                            F.col("end_transcript"))
    
                                                             )
                                .otherwise(F.lit(""))
                                )
                    )
    #databricks-meta-llama-3-1-8b-instruct
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
    # display(df_ai.toPandas())
    

    df_ai_pandas = df_ai.toPandas()
    # df_ai_pandas.to_excel("intermediate_disconnects_appropriately.xlsx")

    # print("LLM hits done",df_ai_pandas.shape )
    log.info(f"LLM hits done {df_ai_pandas.shape}")

    df_ai_pandas['LLM_output_correct_termination'] = df_ai_pandas['ai_response'].fillna("N.A.").astype(str)


    df_ai_pandas['Disconnects the interaction appropriately'] = np.where(    
    df_ai_pandas['LLM_output_correct_termination'].str.strip().str.lower() =='no', "No Evidence", "Full Evidence")

    df_ai_pandas['check_caller'] = df_ai_pandas['transcript_x'].str.contains("CALLER:", na=False , case =True).astype(int)

    df_ai_pandas['Disconnects the interaction appropriately'] = np.where(((df_ai_pandas['Disconnects the interaction appropriately'] =='No Evidence') & (df_ai_pandas['check_caller']==0)), "Full Evidence", df_ai_pandas['Disconnects the interaction appropriately'])


    df_ai_pandas['Disconnects the interaction appropriately'] = np.where(((df_ai_pandas['Disconnects the interaction appropriately'] =='No Evidence') & (df_ai_pandas['transcript_x'].str.contains('voicemail', case=False, na=False))), "Full Evidence", df_ai_pandas['Disconnects the interaction appropriately'])
    df_ai_pandas['Disconnects the interaction appropriately'] = np.where(((df_ai_pandas['Disconnects the interaction appropriately'] =='No Evidence') & (df_ai_pandas['transcript_x'].str.contains('messaging service', case=False, na=False))), "Full Evidence", df_ai_pandas['Disconnects the interaction appropriately'])
    df_ai_pandas['Disconnects the interaction appropriately'] = np.where(((df_ai_pandas['Disconnects the interaction appropriately'] =='No Evidence') & (df_ai_pandas['transcript_x'].str.contains('please leave a message', case=False, na=False))), "Full Evidence", df_ai_pandas['Disconnects the interaction appropriately'])
    df_ai_pandas['Disconnects the interaction appropriately'] = np.where(((df_ai_pandas['Disconnects the interaction appropriately'] =='No Evidence') & (df_ai_pandas['transcript_x'].str.contains("can't answer the phone right now", case=False, na=False))), "Full Evidence", df_ai_pandas['Disconnects the interaction appropriately'])
    df_ai_pandas['Disconnects the interaction appropriately'] = np.where(((df_ai_pandas['Disconnects the interaction appropriately'] =='No Evidence') & (df_ai_pandas['transcript_x'].str.contains('please leave your message', case=False, na=False))), "Full Evidence", df_ai_pandas['Disconnects the interaction appropriately'])
    df_ai_pandas['Disconnects the interaction appropriately'] = np.where(((df_ai_pandas['Disconnects the interaction appropriately'] =='No Evidence') & (df_ai_pandas['transcript_x'].str.contains('disconnecting the call', case=False, na=False))), "Full Evidence", df_ai_pandas['Disconnects the interaction appropriately'])
    df_ai_pandas['Disconnects the interaction appropriately'] = np.where(((df_ai_pandas['Disconnects the interaction appropriately'] =='No Evidence') & (df_ai_pandas['transcript_x'].str.contains('there is no response from your end, i need to disconnect the call', case=False, na=False))), "Full Evidence", df_ai_pandas['Disconnects the interaction appropriately'])
    df_ai_pandas['Disconnects the interaction appropriately'] = np.where(((df_ai_pandas['Disconnects the interaction appropriately'] =='No Evidence') & (df_ai_pandas['transcript_x'].str.contains("there's no response from your end, i need to disconnect the call", case=False, na=False))), "Full Evidence", df_ai_pandas['Disconnects the interaction appropriately'])
    df_ai_pandas['Disconnects the interaction appropriately'] = np.where(((df_ai_pandas['Disconnects the interaction appropriately'] =='No Evidence') & (df_ai_pandas['transcript_x'].str.contains("there's no response from your end, i need to disconnect the call", case=False, na=False))), "Full Evidence", df_ai_pandas['Disconnects the interaction appropriately'])
    df_ai_pandas['Disconnects the interaction appropriately'] = np.where(((df_ai_pandas['Disconnects the interaction appropriately'] =='No Evidence') & (df_ai_pandas['transcript_x'].str.contains("have to disconnect the call", case=False, na=False))), "Full Evidence", df_ai_pandas['Disconnects the interaction appropriately'])

    df_ai_pandas['Disconnects the interaction appropriately'] = np.where(((df_ai_pandas['Disconnects the interaction appropriately'] =='No Evidence') & (df_ai_pandas['transferred'] == 1)), "Full Evidence", df_ai_pandas['Disconnects the interaction appropriately'])

    df_ai_pandas['Disconnects the interaction appropriately'] = np.where(((df_ai_pandas['Disconnects the interaction appropriately'] =='No Evidence') & (df_ai_pandas['transcript_x'].str.contains("hold", case=False, na=False))), "Full Evidence", df_ai_pandas['Disconnects the interaction appropriately'])

    df_ai_pandas['Disconnects the interaction appropriately'] = np.where(((df_ai_pandas['Disconnects the interaction appropriately'] =='No Evidence') & (df_ai_pandas['transcript_x'].str.contains("after the beep", case=False, na=False))), "Full Evidence", df_ai_pandas['Disconnects the interaction appropriately'])

    df_ai_pandas['Disconnects the interaction appropriately'] = np.where(((df_ai_pandas['Disconnects the interaction appropriately'] =='No Evidence') & (df_ai_pandas['transcript_x'].str.contains("refrain from using inappropriate language", case=False, na=False))), "Full Evidence", df_ai_pandas['Disconnects the interaction appropriately'])




    df_ai_pandas['Disconnects the interaction appropriately'] = np.where(((df_ai_pandas['Disconnects the interaction appropriately'] =='No Evidence') & (df_ai_pandas['transcript_x'].str.contains("press", case=False, na=False))), "Full Evidence", df_ai_pandas['Disconnects the interaction appropriately'])

    df_ai_pandas['Disconnects the interaction appropriately'] = np.where(((df_ai_pandas['Disconnects the interaction appropriately'] =='No Evidence') & (df_ai_pandas['transcript_x'].str.contains("can't take your call", case=False, na=False))), "Full Evidence", df_ai_pandas['Disconnects the interaction appropriately'])

    df_ai_pandas['Disconnects the interaction appropriately'] = np.where(((df_ai_pandas['Disconnects the interaction appropriately'] =='No Evidence') & (df_ai_pandas['transcript_x'].str.contains("bye", case=False, na=False))), "Full Evidence", df_ai_pandas['Disconnects the interaction appropriately'])

    df_ai_pandas['Disconnects the interaction appropriately'] = np.where(((df_ai_pandas['Disconnects the interaction appropriately'] =='No Evidence') & (df_ai_pandas['transcript_x'].str.contains("goodbye", case=False, na=False))), "Full Evidence", df_ai_pandas['Disconnects the interaction appropriately'])

    # can't take your call at the moment


    df_ai_pandas['Disconnects the interaction appropriately'] = np.where(((df_ai_pandas['Disconnects the interaction appropriately'] =='No Evidence') & (df_ai_pandas['transcript_x'].str.contains("resolved", case=False, na=False))), "Full Evidence", df_ai_pandas['Disconnects the interaction appropriately'])

    df_ai_pandas['Disconnects the interaction appropriately'] = np.where(((df_ai_pandas['Disconnects the interaction appropriately'] =='No Evidence') & (df_ai_pandas['transcript_x'].str.contains("resolve", case=False, na=False))), "Full Evidence", df_ai_pandas['Disconnects the interaction appropriately'])

    df_ai_pandas['Disconnects the interaction appropriately'] = np.where(df_ai_pandas['eligibility']==0, 'N.A',  df_ai_pandas['Disconnects the interaction appropriately'])

    df_ai_pandas['iAudit Result Critical Question: Disconnects Interaction Appropriately'] = df_ai_pandas['Disconnects the interaction appropriately']

    df_ai_pandas['iAudit Score Critical Question: Disconnects Interaction Appropriately'] = np.where(
      df_ai_pandas['iAudit Result Critical Question: Disconnects Interaction Appropriately'] == 'No Evidence', 0, 10   
     )
    df_ai_pandas['iAudit Comment Critical Question: Disconnects Interaction Appropriately'] = np.where(
    df_ai_pandas['iAudit Result Critical Question: Disconnects Interaction Appropriately'] == 'No Evidence', "Agent terminated the call",  ""   
     )
    
    validate_phrases_list = [
    'voicemail',
    'messaging service',
    'please leave a message',
    "can't answer the phone right now",
    'please leave your message',
    'disconnecting the call',
    'there is no response from your end, i need to disconnect the call',
    "there's no response from your end, i need to disconnect the call",
    "have to disconnect the call", "can i disconnect the call"
     ]
    pattern = "|".join(validate_phrases_list)
    df_ai_pandas['iAudit Confidence Critical Question: Disconnects Interaction Appropriately'] = np.where(df_ai_pandas['iAudit Result Critical Question: Disconnects Interaction Appropriately']== 'Full Evidence', 'High',np.where(df_ai_pandas['transcript_x'].str.contains(pattern, case = False, na = False), 'Low', 'High'))

    df_ai_pandas['iAudit Confidence Critical Question: Disconnects Interaction Appropriately'] = np.where(((df_ai_pandas['iAudit Confidence Critical Question: Disconnects Interaction Appropriately']== 'Low') & (df_ai_pandas['check_caller']==0)), 'High',df_ai_pandas['iAudit Confidence Critical Question: Disconnects Interaction Appropriately'] )


    
    df_ai_pandas.to_excel("intermediate_disconnects_appropriately.xlsx")

    df_ai_pandas_final = df_ai_pandas[['callid','callid_agentid','AgentId','iAudit Score Critical Question: Disconnects Interaction Appropriately','iAudit Result Critical Question: Disconnects Interaction Appropriately','iAudit Comment Critical Question: Disconnects Interaction Appropriately', 'iAudit Confidence Critical Question: Disconnects Interaction Appropriately']]
    
    log.info(f"Final shape: {df_ai_pandas_final.shape}")
    return df_ai_pandas_final

if __name__=="__main__":

    formatted_date ='2026-02-16'
   
    # transcripts = pd.read_excel(r'/Workspace/Users/ananthagiri_abhiram@next.co.uk/iAudit_Trisha/UATpool2CallsRedcatedTranscripts.xlsx')
    # transcripts = pd.read_excel(r'/Workspace/Users/ananthagiri_abhiram@next.co.uk/iAudit_Trisha/20251029102454.xlsx')

    # transcripts = pd.read_excel(r'/Workspace/Users/ananthagiri_abhiram@next.co.uk/iAudit_Trisha/DisconnectsApproriatelyTestTrasncript.xlsx')
   
   
    raw_table_query = f"""SELECT * FROM contactcentre_prod.transcripts.transcripts_raw WHERE DATE(createdate) = '{formatted_date}'"""
   
    full_raw_table = spark.sql(raw_table_query)
    transcripts = full_raw_table.toPandas()

    df_ai_pandas_final = disconnects_appropriately_main_function(transcripts)
    df_ai_pandas_final.to_excel("DisconnectsAppropriatelyFinalResults1602.xlsx")


