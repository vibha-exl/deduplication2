import pandas as pd
import json, re, os, sys, requests
from time import time
import numpy as np
from batch2_utils.understanding_utils import k2, k3, keywords
from prompts.understanding_prompt import acknowledge_prompt, llm_hit2_prompt, apol_conf_prompt
from pyspark.sql.functions import expr
from pyspark.sql import SparkSession
from rapidfuzz import process, fuzz
sys.path.append('../')
from utils.iaudit_logger import get_logger
from utils.iaudit_llm_batch import LLMBatchRequest
import yaml

main_config = yaml.safe_load(open('../main_config.yaml', 'r'))
model_8b = main_config['LLM']['llama']
model_70b = 'contact_centre_internal_batch_large'
model_70b_new = 'databricks-meta-llama-3-3-70b-instruct'


def json_decode(x):
    try:
        x = repair_json(x)
        y = json.loads(x)
        return y
    except Exception as e:
        print("Error decoding json:",x,"\nError is:",e)
        return ""

spark = SparkSession.builder.getOrCreate()
logger = get_logger()

ack_response_format = json.dumps({
    "type": "json_schema",
    "json_schema": {
        "name": "customer_acknowledgement_validation",
        "schema": {
            "type": "object",
            "properties": {
                "agent_paraphrase_customer_query": {
                    "type": "string",
                    "enum": ["yes", "no"],
                    "description": "Whether agent paraphrased customer query."
                },
                "agent_ask_question_relevant_to_query": {
                    "type": "string",
                    "enum": ["yes", "no"],
                    "description": "Whether agent asked question relevant to customer query."
                },
                "agent_say_anything_partly_or_wholly_relevant_to_customers_query": {
                    "type": "string",
                    "enum": ["yes", "no"],
                    "description": "Whether agent say anything partly or wholly relevant to customer query."
                },
                "agent_ask_personal_identification_information": {
                    "type": "string",
                    "enum": ["yes", "no"],
                    "description": "Whether agent collected some personal details."
                },
                "agent_acknowledged_customer_query": {
                    "type": "string",
                    "enum": ["yes", "no"],
                    "description": "Whether agent partly or wholly acknowledged customer query."
                },
                "explain": {
                    "type": "string",
                    "minLength": 1,
                    "description": "Short justification for the decisions above."
                }
            },
            "required": [ "agent_paraphrase_customer_query",
    "agent_say_anything_partly_or_wholly_relevant_to_customers_query",
    "agent_ask_question_relevant_to_query",
    "agent_ask_personal_identification_information",
    "agent_acknowledged_customer_query",
    "explain"
            ],
            "additionalProperties": False
        },"strict": True
    }
})

distress_line_response_format = json.dumps({
    "type": "json_schema",
    "json_schema": {
        "name": "customer_distress_validation",
        "schema": {
            "type": "object",
            "properties": {
                "think_step_by_step": {
                    "type": "string",
                    "minLength": 1,
                    "description": "brief step by step thought process"
                },

                "customers_objective_of_calling": {
                    "type": "string",
                    "minLength": 1,
                    "description": "Customer’s main objective stated in brief."
                },
                "first_line_where_customer_states_query": {
                    "type": "string",
                    "minLength": 1,
                    "description": "Exact first line spoken by customer stating their query."
                },
                "does_the_customers_query_cause_them_any_distress_due_to_the_company": {
                    "type": "string",
                    "enum": ["yes", "no"],
                    "description": "Whether customer shows distress due to company action or delay."
                },
                "line_of_distress_to_customer": {
                    "type": ["string", "null"],
                    "description": "Exact line expressing distress; null if no distress expressed."
                },
                "did_agent_apologize_after_customer_expressing_distress": {
                    "type": "string",
                    "enum": ["yes", "no"],
                    "description": "Whether agent apologized after distress was expressed."
                },
                "did_agent_empathized_after_customer_expressing_distress": {
                    "type": "string",
                    "enum": ["yes", "no"],
                    "description": "Whether agent empathized after distress was expressed."
                },
                "distress_explanation": {
                    "type": "string",
                    "minLength": 1,
                    "description": "Short justification for the decisions above."
                }
            },
            "required": ["think_step_by_step",
                "customers_objective_of_calling",
                "first_line_where_customer_states_query",
                "does_the_customers_query_cause_them_any_distress_due_to_the_company",
                "line_of_distress_to_customer",
                "did_agent_apologize_after_customer_expressing_distress",
                "distress_explanation"
            ],
            "additionalProperties": False
        },"strict": True
    }
})



def merge_consecutive_lines(convo):
    if not convo or convo.strip() == "":
        return ""
    # Split transcript into lines and filter out empty lines
    lines = [line for line in convo.split('\n') if line.strip()]
    if not lines:
        return ""
    merged_lines = []
    current_speaker = None
    current_text = []

    for line in lines:
        # Skip lines that don't contain a colon separator
        if ':' not in line:
            continue
            
        speaker, text = line.split(":", 1)
        speaker = speaker.strip()
        text = text.strip()

        if speaker == current_speaker:
            current_text.append(text)
        else:
            if current_speaker is not None:
                merged_lines.append(f"{current_speaker}: {' '.join(current_text)}")
            current_speaker = speaker
            current_text = [text]

    if current_speaker is not None:
        merged_lines.append(f"{current_speaker}: {' '.join(current_text)}")

    return '\n'.join(merged_lines)


def extraction_llm(df,batch_no, model):
    #columns in df : callid(actually pk), convo
    df['convo_new'] = df['convo'].apply(merge_consecutive_lines)
    df['convo_new'] = df['convo_new'].apply(lambda x: x.replace("AGENT:","Agent:"))
    df['convo_new'] = df['convo_new'].apply(lambda x: x.replace("CALLER:","Customer:"))

    #1st llm hit for acknowledge
    # llama_request = LLMBatchRequest('contact_centre_internal_batch_large')
    # llama_request = LLMBatchRequest(model_70b_new,max_tokens=1024)
    llama_request = LLMBatchRequest(model,max_tokens=1024)
    try:
        extraction_prompt_json = json.dumps(acknowledge_prompt)
        df_spark = spark.createDataFrame(df)
        df = llama_request.apply_batch(
            df_spark,f"concat({extraction_prompt_json}, convo_new)",result_column='ack_llm_result1', response_format = distress_line_response_format
        ).toPandas()
        logger.info(f"First LLM hit for extraction completed : {df.columns}")
        df["ack_llm_result1_parsed"] = df["ack_llm_result1"].apply(json_decode)
    except Exception as e:
        logger.error(f"extraction_llm - acknowledge_llm_1: {e}")
    
    # df.to_excel('/Workspace/Users/ananthagiri_abhiram@next.co.uk/iAudit_Abhiram/Und_customer/intermediate files/extraction_results.xlsx')
    
    #########  parse llm columns
    df['first_line_where_customer_states_query'] = ''
    df['distress_line'] = ''
    df['empathy'] = ''
    df['conv_id'] = df['callid']
    for idx, row in df.iterrows():
        try:
            parsed = row['ack_llm_result1_parsed']
            df.at[idx, 'first_line_where_customer_states_query'] = parsed.get('first_line_where_customer_states_query', '')
            df.at[idx, 'distress_line'] = parsed.get('line_of_distress_to_customer', '')
            df.at[idx, 'empathy'] = parsed.get('did_agent_empathized_after_customer_expressing_distress', '')
        except Exception as e:
            logger.error(f"extraction_llm - parsing_ack_llm_1 at row {idx}: {e}")
            logger.error(f"error in extracting the output at row {idx}: {row['ack_llm_result1_parsed']}")
    return df

def contains_keywords(line):
    # Convert the input line to lowercase for case-insensitive comparison
    line = str(line)
    line_lower = line.lower()
    if "let" in line:
        if "look" in line:
            return True
    # Check if any keyword is present in the line (understanding_utils keywords list)
    keywords2 = set(keywords)
    for keyword in keywords2:
        if keyword in line_lower:
            return True
    return False
 
def process_lines(lines, best_batch,batch_no):
    try:
        # Initialize variables
        before_best_batch = []
        next_line_after_best_batch = None
        after_best_batch = []
        er = False
        # Find the index of the line that matches best_batch
        try:
            index_best_batch = lines.index(best_batch)
        except:
            er=True
            return ["no_match"],"no_match",["no_match"]
            
        # Split the list into parts
        before_best_batch = lines[:index_best_batch]
        if index_best_batch + 1 < len(lines):
            next_line_after_best_batch = lines[index_best_batch + 1]
            after_best_batch = lines[index_best_batch + 1:]
            b2=0
            for i2 in after_best_batch:
                # i2 = i2.lower()
                if "AGENT" in i2:
                    if len(next_line_after_best_batch.split(" "))<40:
                        next_line_after_best_batch = (next_line_after_best_batch) +" "+i2[7:]

                    b2+=1
                if b2 == 5:
                    break
        else:
            after_best_batch = []
            if index_best_batch + 4 < len(lines):
                n2 = lines[index_best_batch + 3][7:]
                if len(next_line_after_best_batch.split(" "))<15:
                    next_line_after_best_batch = (next_line_after_best_batch) +" "+n2
    
        return before_best_batch, next_line_after_best_batch, after_best_batch
    except Exception as e:
        logger.error(f"Acknowledge - process_lines: {e}")
        raise
        
def match_loop(df4, batch_no):
    df4["next_line_to_query"] = ""
    df4["query_line"] = ""
    df4["apology_evidence"] = ""
    df4["apology"] = ""
    
    try:
        for a in range(len(df4)):
            aa = df4["convo"][a]
            aa = merge_consecutive_lines(aa)
            lines = aa.split("\n")
            lines = [line for line in lines if line.strip()]

            t = df4["first_line_where_customer_states_query"][a]
            lines2 = []
            for line in lines:
                if "CALLER:" in line:
                    lines2.append(line)
            
            # Find best matching customer query line using word overlap
            max_common_count = 0
            best_line = "nothing"
            for line in lines2:
                words1 = set(t.lower().split())
                words2 = set(line.lower().split())
                common_count = len(words1.intersection(words2))
                if common_count > max_common_count:
                    max_common_count = common_count
                    best_line = line
            
            best_match = best_line
            
            if best_match == "nothing":
                continue
            
            # Get lines before/after the query line
            before_best_batch, next_line_after_best_batch, after_best_batch = process_lines(lines, best_match, batch_no)
            
            if next_line_after_best_batch == "no_match":
                df4.loc[a, "query_line"] = "no match"
                df4.loc[a, "next_line_to_query"] = "no match"
                df4.loc[a, "apology_evidence"] = "no match"
                df4.loc[a, "apology"] = "yes"
                continue

            # Check for apology in next 9 agent lines after query
            b = 0
            ap = False
            for afterline in after_best_batch:
                if "AGENT:" in afterline:
                    af2 = afterline.lower()
                    if ("sorry" in af2) or ("apolog" in af2):
                        df4.loc[a, "apology_evidence"] = afterline
                        df4.loc[a, "apology"] = "yes"
                        ap = True
                        break
                    b += 1
                if b == 9:
                    break
            
            if ap == False:
                df4.loc[a, "apology"] = "no"
                df4.loc[a, "apology_evidence"] = "none"
            else:
                df4.loc[a, "apology"] = "yes"
                df4.loc[a, "apology_evidence"] = afterline
            
            df4.loc[a, "query_line"] = best_match
            df4.loc[a, "next_line_to_query"] = next_line_after_best_batch
        
        return df4
    except Exception as e:
        logger.error(f"Acknowledge - match_loop: {e}")
        raise


def llm_ack_check(df,batch_no, model):
    #1st llm hit for acknowledge
    llama_request = LLMBatchRequest(model,max_tokens=1024, temperature=0.1)
    df['ack_llm_result2'] = '''{
    "agent_paraphrase_customer_query":"yes",
    "agent_say_anything_partly_or_wholly_relevant_to_customers_query":"yes",
    "agent_ask_question_relevant_to_query":"yes",
    "agent_ask_personal_identification_information":"",
    "agent_acknowledged_customer_query":"yes",
    "explain":"default keyword check"
    }
    '''
    ack_mask = (
        ~df['next_line_to_query'].apply(contains_keywords)
        & df['query_line'].notnull() & (df['query_line'] != "")
        & df['next_line_to_query'].notnull() & (df['next_line_to_query'] != "")
    )
    if ack_mask.any():
        try:
            ack_prompt_json = json.dumps(llm_hit2_prompt)
            df_spark = spark.createDataFrame(df[ack_mask])
            result_df = llama_request.apply_batch(df_spark, f"concat({ack_prompt_json}, query_line, '\n### Response: ', next_line_to_query)", result_column='ack_llm', response_format = ack_response_format ).toPandas()
            df.loc[ack_mask, 'ack_llm_result2'] = result_df['ack_llm'].values
            # df["ack_llm_result2_parsed"] = df["ack_llm_result2"].apply(json_decode)
            # df = pd.concat([df, pd.json_normalize(df['ack_llm_result2_parsed'])], axis=1)

        except Exception as e:
            logger.error(f"Acknowledge - acknowledge_llm_2: {e}")
        df["ack_llm_result2_parsed"] = df["ack_llm_result2"].apply(json_decode)
        df = pd.concat([df, pd.json_normalize(df['ack_llm_result2_parsed'])], axis=1)
        return df
    else:
        df["ack_llm_result2_parsed"] = df["ack_llm_result2"].apply(json_decode)
        df = pd.concat([df, pd.json_normalize(df['ack_llm_result2_parsed'])], axis=1)
        return df




def get_actual_distress_and_next_lines(row):
    next_n = 9
    lines = merge_consecutive_lines(row['convo']).split("\n")

    matches = process.extract(
        row["distress_line"],
        lines,
        scorer=fuzz.partial_ratio,
        limit=None
    )
    # Get all lines with similarity > 80
    top_lines = [match[0] for match in matches if match[1] > 80]
    if top_lines:
        for line in top_lines:
            if "CALLER" in line:
                actual_distress = line
                # For the first matched line, get next 5 lines
                idx = lines.index(top_lines[0])
                next_lines = lines[idx+1:idx+1+next_n]
                prior_lines = lines[:idx]
                next_lines_str = "\n".join(next_lines)
                prior_lines_str = "\n".join(prior_lines)
                if len(next_lines_str.split())<50:
                    next_lines = lines[idx+1:idx+2+next_n]
                    next_lines_str = "\n".join(next_lines)
                break
            else:
                actual_distress = ""
                next_lines_str = ""
                prior_lines_str = ""
    else:
        actual_distress = ""
        next_lines_str = ""
        prior_lines_str = ""
    return pd.Series([actual_distress, next_lines_str, prior_lines_str])
    
def parse_apol_llm_result(row):
        try:
            return pd.Series(row['apol_llm_result'])
        except Exception:
            return pd.Series() 
        
def run_apology_conf_llm(ackdf,model= model_70b_new):     # model='contact_centre_internal_batch_large'
    dfs = ackdf[ackdf['Actual distress line'] !=  '']
    sparkdf = spark.createDataFrame(dfs[['conv_id','Actual distress line', 'Next 5 lines']])
    llama_request = LLMBatchRequest(model, temperature=0.1, max_tokens=1024)
    json_prompt = json.dumps(apol_conf_prompt)
    result_df = llama_request.apply_batch(sparkdf, f"concat({json_prompt}, `Actual distress line`,'\n', `Next 5 lines`)",result_column="apol_llm_result")
    result_dfpd = result_df.toPandas()[['conv_id','apol_llm_result']]
    if len(dfs)>0:
        merged_df = ackdf.merge(result_dfpd, on='conv_id', how='left')
    else:
        merged_df =  ackdf.copy()
    merged_df['apol_llm_result'] = merged_df['apol_llm_result'].apply(lambda x: json_decode(x) if pd.notnull(x) else x)
    dff = merged_df.join(merged_df.copy().apply(parse_apol_llm_result, axis=1)).copy()
    no_distress_mask = (dff['Actual distress line']== '') | (dff['Actual distress line'].isnull())
    dff.loc[no_distress_mask,'label'] = 'Apology not required'
    return dff



def acknowledge_main(df,batch_no, model):
    logger.info("Understanding - Acknowledgement module is running !!")
    try:
        s_t = time()
        ##### merge utterances to conversations
        df = df[['callid','channel','transcript']].drop_duplicates()
        df['convo'] = df['channel'] + ": " + df['transcript']
        df =df.groupby('callid')['convo'].apply(lambda x: "\n".join(x)).reset_index()
        df.columns = ['callid', 'convo']

        #### drop conversations where the length is smaller than 15 lines
        indices_to_drop = []
        for index, row in df.iterrows():
            if len(row['convo'].split('\n')) < 15:
                indices_to_drop.append(index)
        df = df.drop(indices_to_drop)
        df.reset_index(drop=True, inplace=True)
        # logger.info(f"Columns in df after merging: {df.columns}")

        ### first llm hit for acknowledgement, distress line
        df2 = extraction_llm(df.copy(),batch_no, model_8b)
        # logger.info(f"Columns in df2 after first extraction hit: {df2.columns}")

        ### find the actual line in convo using the distress line from llm
        df2[["Actual distress line", "Next 5 lines", "Lines before distress"]] = df2.apply(get_actual_distress_and_next_lines, axis=1)

        ### second llm hit for apology/confidence
        df2 = run_apology_conf_llm(df2,model_70b_new)
        # logger.info(f"Columns in df2 after apology_conf llm : {df2.columns}")


        # df3 = pd.merge(df,df2,on="convo",how="inner")
        # logger.info(f"Columns in df3 after merging with df2 : {df3.columns}")

        df4 = match_loop(df2,batch_no)
        # logger.info(f"Columns in df3 after match_loop : {df4.columns}")

        df5 = llm_ack_check(df4,batch_no, model_8b)
        # logger.info(f"Columns in df3 after llm_ack_check : {df5.columns}")
                
        # df5 = pd.merge(df5,df4,on="query_line",how="right")
        # logger.info(f"Columns in df3 after merging with df5 : {df5.columns}")

        if df5 is None or df5.empty:
            logger.warning("acknowledge_main: df5 is empty or None, returning empty DataFrame.")
            return pd.DataFrame()
        df5["acknowledge"]=""
        df5["apologised_before_distress"] = ""
        for i in range(len(df5)):
            a = df5["agent_paraphrase_customer_query"][i]
            b = df5["agent_say_anything_partly_or_wholly_relevant_to_customers_query"][i]
            c = df5["agent_ask_question_relevant_to_query"][i]
            d = df5["agent_ask_personal_identification_information"][i]
            f = df5["agent_acknowledged_customer_query"][i]

            a = str(a)
            b = str(b)
            c = str(c)
            d = str(d)
            f = str(f)

             ## New logic- Overwrites all previous apology detections 

            e = str(df5["label"][i])
            if e.lower() == "apology not required":
                df5.loc[i,"apology_needed"]="no"
                df5.loc[i,"apology"]="not needed"
            elif  e.lower() == 'apology required but not apologized':
                distress_line = str(df5.loc[i,"Actual distress line"]).lower()
                afl =  str(df5.loc[i,"Next 5 lines"]).lower()
                if len(str(distress_line).lower().split()) <= 5:
                    df5.loc[i,"apology_needed"]="no"
                    df5.loc[i,"apology"]="not needed"
                elif ("sorry" in afl) or ("apolog" in afl):
                    df5.loc[i,"apology_needed"]="yes"
                    df5.loc[i,"apology"]="yes"
                    df5.loc[i,"apology_evidence"] = df5.loc[i,"Actual distress line"] + "\n" + df5.loc[i,"Next 5 lines"]
                else: 
                    previous_lines_to_distress = str(df5.loc[i,"Lines before distress"])
                    prev_al = "\n".join([l for l in previous_lines_to_distress.split("\n") if str(l).strip().lower().startswith("agent")])
                    if ("sorry" in prev_al) or ("apolog" in prev_al):
                        df5.loc[i,"apology_needed"]="yes"
                        df5.loc[i,"apology"]="yes"
                        df5.loc[i,"apology_evidence"] = df5.loc[i,"Lines before distress"]
                        df5.loc[i, "apologised_before_distress"] = 'yes'
                    else:
                        df5.loc[i,"apology_needed"]="yes"
                        df5.loc[i,"apology"]="no"
                        df5.loc[i,"apology_evidence"] = df5.loc[i,"Actual distress line"] + "\n" + df5.loc[i,"Next 5 lines"]
            elif e.lower() == 'apologized':
                df5.loc[i,"apology_needed"]="yes"
                df5.loc[i,"apology"]="yes"
                df5.loc[i,"apology_evidence"] = df5.loc[i,"Actual distress line"] + "\n" + df5.loc[i,"Next 5 lines"]
            else:
                logger.warning(f"Expected Apology label not found for index {i}: {e}")
                df5.loc[i,"apology_needed"]="no"
                df5.loc[i,"apology"]="not needed"

            ###
            
            if df5["apology_needed"][i]=="no":
                #df5["apology"][i]
                df5.loc[i,"apology"]="not needed"
                #df5["apology_evidence"][i]
                df5.loc[i,"apology_evidence"]="none"
            
            if a == "nan":
                df5.loc[i,"acknowledge"] = "yes"
            elif a == "yes":
                df5.loc[i,"acknowledge"] = "yes"
            elif b =="yes":
                df5.loc[i,"acknowledge"] = "yes"
            elif c=="yes":
                df5.loc[i,"acknowledge"] = "yes"
            elif f=="yes":
                df5.loc[i,"acknowledge"] = "yes"
            else :
                df5.loc[i,"acknowledge"] = "no"
        ### save intermediate results
        # df5.to_excel('ack_llm_JanAuditedCallsIntermediate.xlsx')
        # logger.info("Saved Ack intermediate results")

        # df5 = df5[["callid","query_line","apology_needed","next_line_to_query","apology_evidence","apology","acknowledge","label"]]

        logger.info("Understanding - Acknowledge is completed !!")
        return df5
    
    except Exception as e:
        logger.error(f"Understanding - acknowledge_main: {e}")
        raise
   