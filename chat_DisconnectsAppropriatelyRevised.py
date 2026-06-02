# !pip install PyYAML
# !pip install openpyxl

import os,sys
sys.path.append('/Workspace/Users/pawan_kumar@next.co.uk/iAudit_deployment')
sys.path.append('/Workspace/Users/ananthagiri_abhiram@next.co.uk/iAudit_vibha')
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

#########################################################################################################  

    data_req = df_transcripts.copy()

    data_req['transcript_x'] = data_req['transcript']

    data_req['agent_order'] = 1
    data_req['agent_order_max'] = 1
    data_req['callid_agentid'] = data_req['callid']

    #########################################################################################################
    # data_req['createdate'] = pd.to_datetime(data_req['createdate'], format = 'mixed', errors='coerce')
    data_req['createdate'] = pd.to_datetime(data_req['createdate'], utc=True, errors='coerce')
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

    data_req.to_excel("intermediate_disconnects_appropriately.xlsx")



    spark_disconnect_type =  spark.sql(f"""
     SELECT * FROM `contactcentre_prod`.`iaudit`.`vw_disconnect_direction_acw` 
            WHERE conversationId IN {tuple(audit_callids)}      
        """)

    df_disconnect_type = spark_disconnect_type.toPandas()

##################################################################################################

    print("Raw DisconnectType values from vw_disconnect_direction_acw")

    if 'DisconnectType' in df_disconnect_type.columns:
        print(df_disconnect_type['DisconnectType'].value_counts(dropna=False))

    df_disconnect_type.to_excel(
        "Raw_Disconnect_Type_Table.xlsx",
        index=False
    )
#####################################################################################################

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
    # data_req['eligibility'] = np.where(

    #   ((data_req['DisconnectType'] != 'N.A.') & (data_req["agent_order_max"]==data_req["agent_order"] ) & (data_req['DisconnectType'].isin(['client']))), 1,0
    # )


    chat_disconnect_patterns = (
        "advisor has left the conversation|"
        "pick up where we left off|"
        "thank you for contacting next|"
        "pleasure assisting you today|"
        "have a great day ahead|"
        "have a good evening|"
        "anything else i can help with|"
        "hope that answers|"
        "hope that resolves|"
        "offline for a while"
    )

    data_req['eligibility'] = np.where(
        data_req['transcript_x'].str.contains(
            chat_disconnect_patterns,
            case=False,
            na=False
        ),
        1,
        0
    )
########################################################################################################

# ==========================================================
# DEBUG BLOCK FOR DISCONNECT ELIGIBILITY
# ==========================================================

    print("\n================ DISCONNECT DEBUG =================\n")

    print("DisconnectType Distribution")
    print(data_req['DisconnectType'].value_counts(dropna=False))

    print("\nEligibility Distribution")
    print(data_req['eligibility'].value_counts(dropna=False))

    print("\nAgent Order Check")
    print(
        data_req[['agent_order','agent_order_max']]
        .drop_duplicates()
        .head(20)
    )

    # Exact reason why eligibility became 0
    data_req['eligibility_reason'] = np.select(
        [
            data_req['DisconnectType'].isna(),
            data_req['DisconnectType'].eq('N.A.'),
            ~data_req['DisconnectType'].isin(['client']),
            data_req['agent_order_max'] != data_req['agent_order']
        ],
        [
            'DisconnectType Null',
            'DisconnectType N.A.',
            'DisconnectType Not Client',
            'Not Last Agent'
        ],
        default='Eligible'
    )

    print("\nEligibility Reason Distribution")
    print(data_req['eligibility_reason'].value_counts(dropna=False))

    # Export all rows
    debug_cols = [
        'callid',
        'callid_agentid',
        'AgentId',
        'DisconnectType',
        'agent_order',
        'agent_order_max',
        'eligibility',
        'eligibility_reason',
        'transcript_x'
    ]

    data_req[debug_cols].to_excel(
        "DisconnectEligibilityDebug.xlsx",
        index=False
    )

    # Export only eligibility = 1 rows
    data_req[data_req['eligibility'] == 1][debug_cols].to_excel(
        "DisconnectEligibility1.xlsx",
        index=False
    )

    # Export only eligibility = 0 rows
    data_req[data_req['eligibility'] == 0][debug_cols].to_excel(
        "DisconnectEligibility0.xlsx",
        index=False
    )

    print("\nEligibility=1 Count :",
        len(data_req[data_req['eligibility'] == 1]))

    print("Eligibility=0 Count :",
        len(data_req[data_req['eligibility'] == 0]))

    print("\n================ DEBUG COMPLETE =================\n")



    data_req[
    [
    'callid',
    'callid_agentid',
    'AgentId',
    'DisconnectType',
    'agent_order',
    'agent_order_max',
    'eligibility',
    'transcript_x'
    ]
    ].to_excel(
        "DisconnectEligibilityBusinessView.xlsx",
        index=False
    )

    print(data_req['DisconnectType'].value_counts(dropna=False))
    print(data_req['eligibility'].value_counts(dropna=False))
    # ==========================================================
##############################################################################################################



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
                        """ai_query('databricks-gpt-oss-120b',request => final_prompt)"""
                            )
                    
                )
            )
            )
    # display(df_ai.toPandas())
    

    df_ai_pandas = df_ai.toPandas()
    df_ai_pandas.to_excel("2intermediate_disconnects_appropriately.xlsx")

    # print("LLM hits done",df_ai_pandas.shape )
    log.info(f"LLM hits done {df_ai_pandas.shape}")

    df_ai_pandas['LLM_output_correct_termination'] = df_ai_pandas['ai_response'].fillna("N.A.").astype(str)


    df_ai_pandas['Disconnects the interaction appropriately'] = np.where(    
    df_ai_pandas['LLM_output_correct_termination'].str.strip().str.lower() =='no', "No Evidence", "Full Evidence")

    # df_ai_pandas['check_caller'] = df_ai_pandas['transcript_x'].str.contains("CALLER:", na=False , case =True).astype(int)
    df_ai_pandas['check_caller'] = df_ai_pandas['transcript_x'].str.contains(
        "Customer:",
        na=False,
        case=False
    ).astype(int)

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
###################################################################################################################
    df_ai_pandas['Disconnects the interaction appropriately'] = np.where(
        (
            (df_ai_pandas['Disconnects the interaction appropriately'] == 'No Evidence')
            &
            (
                df_ai_pandas['transcript_x'].str.contains(
                    "advisor has left the conversation",
                    case=False,
                    na=False
                )
            )
        ),
        "Full Evidence",
        df_ai_pandas['Disconnects the interaction appropriately']
    )

    df_ai_pandas['Disconnects the interaction appropriately'] = np.where(
        (
            (df_ai_pandas['Disconnects the interaction appropriately'] == 'No Evidence')
            &
            (
                df_ai_pandas['transcript_x'].str.contains(
                    "thank you for contacting next",
                    case=False,
                    na=False
                )
            )
        ),
        "Full Evidence",
        df_ai_pandas['Disconnects the interaction appropriately']
    )

    df_ai_pandas['Disconnects the interaction appropriately'] = np.where(
        (
            (df_ai_pandas['Disconnects the interaction appropriately'] == 'No Evidence')
            &
            (
                df_ai_pandas['transcript_x'].str.contains(
                    "pleasure assisting you today",
                    case=False,
                    na=False
                )
            )
        ),
        "Full Evidence",
        df_ai_pandas['Disconnects the interaction appropriately']
    )
###############################################################################################################
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


    
    df_ai_pandas.to_excel("3intermediate_disconnects_appropriately.xlsx")

    df_ai_pandas_final = df_ai_pandas[['callid','callid_agentid','AgentId','iAudit Score Critical Question: Disconnects Interaction Appropriately','iAudit Result Critical Question: Disconnects Interaction Appropriately','iAudit Comment Critical Question: Disconnects Interaction Appropriately', 'iAudit Confidence Critical Question: Disconnects Interaction Appropriately']]
    
    log.info(f"Final shape: {df_ai_pandas_final.shape}")
    return df_ai_pandas_final

if __name__=="__main__":

    formatted_date = '2026-04-26'

    INPUT_FILE_PATH = spark.table(
        'contactcentre_prod.iaudit.filtered_input_data_iaudit_chat_v2'
    ).filter(
        F.col('call_date') == formatted_date
    )

    transcripts = INPUT_FILE_PATH.toPandas()

    transcripts = transcripts.rename(columns={
        'conversation_id': 'callid',
        'agent_id': 'AgentId',
        'convo': 'transcript',
        'call_date': 'createdate'
    })

    transcripts['channel'] = 'chat'

    df_ai_pandas_final = disconnects_appropriately_main_function(transcripts)
    df_ai_pandas_final.to_excel("DisconnectsAppropriatelyFinalResults1602.xlsx")


