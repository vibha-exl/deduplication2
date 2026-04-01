import sys
import re
import pandas as pd
from uc_acknowledge import acknowledge_main
from uc_repetition import c_rep_main
from pyspark.sql.functions import lit, col
import warnings
import yaml
sys.path.append('../')
from utils.iaudit_logger import get_logger
warnings.filterwarnings("ignore")

main_config = yaml.safe_load(open('../main_config.yaml', 'r'))
model = main_config['LLM']['llama']

log = get_logger()


def split_calls_agentwise(data, log):
    try:
        groups = data.groupby('callid')
        updated_dfs = []
        for callid, df in groups:
            pk = 'callid'
            df = df.reset_index(drop=True)
            df['actual_callid'] = df['callid']
            df[pk] = df['callid'] + "|" + df['AgentId']
            if len(df[pk].unique())>1:
                pk_precedence = df[pk].unique()
                ordered_dfs = []

                for pki in pk_precedence:
                    tempdf = df[df[pk]==pki]
                    ordered_dfs.append(tempdf)
                df = pd.concat(ordered_dfs).reset_index(drop=True)

                df['agent_sequence_number'] = 1  # initialize
                df['queue_sequence_number'] = 1 
                for i in range(len(df)):
                    if i<1:
                        continue
                    if df.at[i, pk] != df.at[i-1, pk]:
                        df.loc[i:,'agent_sequence_number'] = df.at[i-1,'agent_sequence_number'] + 1
                    if df.at[i, 'queue'] != df.at[i-1, 'queue']:
                        df.loc[i:,'queue_sequence_number'] = df.at[i-1,'queue_sequence_number'] + 1
                df['no_of_agents'] = df['agent_sequence_number'].nunique()
                updated_dfs.append(df)
            else:
                df['agent_sequence_number'] = 1
                df['queue_sequence_number'] = 1
                df['no_of_agents'] = 1
                updated_dfs.append(df)
        dfcomb = pd.concat(updated_dfs)
        return dfcomb
    except Exception as e:
        log.error(f"Error in understanding module - split_calls_agentwise: {e}")
        raise

def und_confidence(row):
    
    row['Confidence'] = 'High'
    if row['Score'] == 20:
        return row
    try:
        agent_utterances = ' '.join([line for line in row['convo'].split('\n') if line.strip().startswith(('Agent:', 'agent:'))])
        if row['apology']!= 'no':
            #markdown not due to apology
            row['Confidence'] = 'Low'
        elif 'apolog' in agent_utterances.lower() or 'sorry' in agent_utterances.lower():
            #keyword check in any agent utterance
            row['Confidence'] = 'Low'
        elif str(row['Final_classification']).lower() not in ['eod','complaint']:
            #considering only eod and complaint
            row['Confidence'] = 'Low'
        elif len(row['Next 5 lines'].split('\n'))<5:
            # handling incomplete/doubtful transcripts
            row['Confidence'] = 'Low'
        else:
            pass
    except Exception as e:
        log.info(f"Columns in row: {row.keys()}")
        row['Confidence'] = 'Low'
        log.error(f"Error in understanding module - und_confidence: {e}")
        raise
    return row

def und_score(r12,ack,batch_no,cid, log):
    try:
        df2 = pd.merge(r12, ack.drop(columns=['convo']), on="callid", how="inner")
        df2["Result"]=""
        df2["Score"]=""
        df2["Associated Transcript with Timestamp"]=""


        df2 = df2.drop_duplicates(subset='callid', keep='first')
        df2.reset_index(drop=True, inplace=True)

        for i in range(len(df2)):
            result = "yes"
            evidence = ""
            apology = str(df2["apology"][i])
            repeat = str(df2["repeat"][i])
            repeat = repeat.strip()
            nl = str(df2["next_line_to_query"][i])
            acknowledge = df2["acknowledge"][i]
            apol_evidence = df2["apology_evidence"][i]
            # eod = str(df2["eod"][i])
            # if "fail" in eod:
            #     result = "yes"
            if apology == "no" and acknowledge == "no":
                result = "no"
                # evidence = evidence + "Apology and Acknowledgement for Customer's query missing : \n"+df2["query_line"][i]+"\n"+nl+"\n"
                evidence = evidence + "Acknowledgement for Customer's query missing : \n"+df2["query_line"][i]+"\n"+nl+"\n"
                evidence = evidence + "Apology/Empathy for customers query is missing : \n"+ apol_evidence +"\n"
            elif apology == "no":
                result = "no"
                # evidence = evidence + "Apology for Customer's query missing : \n"+df2["query_line"][i]+"\n"+nl+"\n"
                evidence = evidence + "Apology/Empathy for customers query is missing : \n"+ apol_evidence +"\n"
            elif acknowledge == "no":
                result = "no"
                evidence = evidence + "Acknowledgement for Customer's query missing : \n"+df2["query_line"][i]+"\n"+nl+"\n"
         
            if nl == "nan":
                result = "yes"
                evidence = ""
            
            if repeat == "yes":
                result = "no"
                # evidence = evidence + "repetition detected : \nline 1 : "+df2["l1"][i]+"\nline2 : "+df2["l2"][i]
                evidence = evidence + "repetition detected : \n" + df2["conversation_chunks"][i]
                if df2["multiple_rep"][i]=="yes":
                    evidence = evidence + "\nMore repetitions available in full transcript"
            
            df2.loc[i,"Result"]=result
            if result == "no":
                df2.loc[i,"Score"]=0
            else:
                df2.loc[i,"Score"]=20
            
            df2.loc[i,"Associated Transcript with Timestamp"]=evidence
            
        # df2=df2[["callid","Result","Confidence","Score","Associated Transcript with Timestamp"]]
        cid2 = df2["callid"].unique()
        for i in cid:
            i = i.strip()
            if i not in cid2:
                new_row = {"callid":i,"Result":"yes","Score":20,"Associated Transcript with Timestamp":""}
                new_row = pd.DataFrame([new_row])
                
                df2 = pd.concat([df2, new_row], ignore_index=True)
                df2.reset_index(drop=True, inplace=True)
        
        return df2
    except Exception as e:
        log.error(f"Error in understanding module - und_score: {e}")
        raise

def understanding_customer_main(df, batch_no, log):
    try:
        df = split_calls_agentwise(df, log)
        ack = acknowledge_main(df,batch_no, model)
        r12 = c_rep_main(df,batch_no, model)
        und = und_score(r12,ack,batch_no,df["callid"].unique(), log)
        und['pk'] = und['callid']
        ids = und['pk'].unique().tolist()

        ###### pull eod data
        df_spark_eod = spark.sql(f"""
                SELECT pk, Final_classification
                FROM contactcentre_prod.iaudit.intermediate_eod_v2_complaints_results
                WHERE pk IN {tuple(ids)}
            """)
        df_pd_eod = df_spark_eod.toPandas()
        ###### add confidence score
        und = und.merge(df_pd_eod, on='pk', how='left')

        und = und.apply(und_confidence, axis=1)
        und['callid'] = und['pk'].apply(lambda x: x.split('|')[0])
        und['agentId'] = und['pk'].apply(lambda x: x.split('|')[1])
        return und
    except Exception as e:
        log.error(f"Error in understanding module - understanding_customer_main: {e}")
        raise


if __name__=="__main__":
    log = get_logger()
    log.info("Understanding Module is running !!")
    try:
        catalog_config = yaml.safe_load(open('../catalog_config.yaml', 'r'))
        try:
            formatted_date = dbutils.jobs.taskValues.get(taskKey="Pre-Modules-Run", key="sql_date")
        except Exception as e:
            formatted_date = sys.argv[1]
            if not re.match(r"^\d{4}-\d{2}-\d{2}$", str(formatted_date)):
                raise ValueError(f"formatted_date '{formatted_date}' is not in yyyy-mm-dd format")
            logger.info(f"formatted_date: {formatted_date}")

        INPUT_FILE_PATH = spark.table(catalog_config['output_table']['filtered_table']).filter(f"call_date = '{formatted_date}'")
        df_transcripts1 = INPUT_FILE_PATH.toPandas()

        df1 = df_transcripts1.copy()
        
        total_calls = len(df_transcripts1['callid'].unique().tolist())
        # filtered_calls = df_transcripts1['callid'].unique().tolist()[:total_calls//2]

        filtered_calls = df_transcripts1['callid'].unique().tolist()

        df = df1[df1['callid'].isin(filtered_calls)].reset_index(drop=True)
        if 'index' not in df.columns:
            df['index'] = 1
        batch_no =1
        callids = df['callid'].unique().tolist()
        batch_size = 2000
        df_batches = [df[df['callid'].isin(callids[i:i+batch_size])].reset_index(drop=True) for i in range(0, len(callids), batch_size)]
        result_dfs = []
        for batch_no,eachdf in enumerate(df_batches):
            batch_result = understanding_customer_main(eachdf, batch_no, log)
            if batch_result.empty:
                continue
            result_dfs.append(batch_result)
        final_output = pd.concat(result_dfs)
        log.info(
                f"Completed understanding customer Run :{final_output['Score'].value_counts()}, "
                f"Confidence score added : {final_output['Confidence'].value_counts()}, "
            )
        try:
            final_output.to_excel(f"/Workspace/Users/ananthagiri_abhiram@next.co.uk/iAudit_Abhiram/Und_customer/intermediate files/{formatted_date}_uc_IntermediateResults.xlsx")
        except Exception as e:
            log.error("Error Saving intermediate outputs- {e}")

        final_columns = ['callid','Result','Score','Associated Transcript with Timestamp','Confidence','pk','agentId']
        final_output = final_output[final_columns]


        #Garbage collection
        del final_output
        del df_batches
        del df
        del df1
        del df_transcripts1

    except Exception as e:
        log.error(f"Error in understanding module: {e}")
        raise