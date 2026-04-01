import pandas as pd
import json
from json_repair import repair_json
from time import time
import os,sys
import requests
import numpy as np
from prompts.understanding_prompt import repetition_prompt, rep_confirmation_prompt
from pyspark.sql.functions import expr
import re
sys.path.append('../')
from utils.iaudit_logger import get_logger
from utils.iaudit_llm_batch import LLMBatchRequest
from pyspark.sql import SparkSession
from pyspark.dbutils import DBUtils
import yaml

logger = get_logger()

spark = SparkSession.builder.getOrCreate()
dbutils = DBUtils(spark)
DATABRICKS_TOKEN = dbutils.notebook.entry_point.getDbutils().notebook().getContext().apiToken().get()

# model = dbutils.jobs.taskValues.get(taskKey="Pre-Modules-Run", key="llama")
main_config = yaml.safe_load(open('../main_config.yaml', 'r'))
model = main_config['LLM']['llama']
# def json_decode(x):
#     try:
#         x = repair_json(x)
#         y = json.loads(x)
#         return y
#     except Exception as e:
#         raise f"Repetition - json_decode: {e}"
#         return ""
    
def merge_consecutive_lines(lines):
    if not lines:
        return []

    merged_lines = []
    current_speaker = None
    current_text = []

    for line in lines:
        #print(line)
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

    return merged_lines

#batch_no = 800
def concatenate_strings(series):
    return '\n'.join(series)

def get_cosine_similarity(lines):

    dataset = pd.DataFrame({
    "sentences": [
        lines
    ]
    })

    def create_tf_serving_json(data):
        return {'inputs': {name: data[name].tolist() for name in data.keys()} if isinstance(data, dict) else data.tolist()}

    try:

        url = "https://adb-6188831950334199.19.azuredatabricks.net/serving-endpoints/get_cosine_sim/invocations"
        # print('url:',url)
        headers = {'Authorization': f'Bearer {DATABRICKS_TOKEN}', 'Content-Type': 'application/json'}
        ds_dict = {'dataframe_split': dataset.to_dict(orient='split')} if isinstance(dataset, pd.DataFrame) else create_tf_serving_json(dataset)
        data_json = json.dumps(ds_dict, allow_nan=True)
        response = requests.request(method='POST', headers=headers, url=url, data=data_json)
        if response.status_code != 200:
            raise Exception(f'Request failed with status {response.status_code}, {response.text}')
        output = response.json()
        return [list(x.values()) for x in output['predictions']]

    except Exception as e:
        logger.error(f"Repetition - get_cosine_similarity: {e}\n No of elements in input list: {len(lines)}")
        raise
        # return []
        
def embed_hit(df,batch_no):
    try:
        l12=[]
        for i in range(len(df)):
            aa = df["convo"][i]
            convo = aa
            cust=[]
            
            lines = aa.split("\n")
            lines = [line for line in lines if line.strip()]
            lines = merge_consecutive_lines(lines)
            
            speaker = []
            for line in lines:
                #print(line)
                if "CALLER:" in line[:10]:
                    line = line.split(": ")
                    # spk = line[0]
                    line = line[1]
                    if len(line.split(" "))<=5:
                        continue
                    elif (len(line.split(" "))<10): 
                        # if ("thank" in line.lower()) or "yes" in line.lower() or "yeah" in line.lower() or "ok" in line.lower():
                        if ("thank" in line.lower()):
                            continue
                    
                    elif "thank" in line.lower():
                        if len(line.split(" "))<15:
                            continue
                     
                    elif "press one" in line.lower() or "press *" in line.lower(): #Remove IVR
                        continue
                    else:
                        dict2 = {"line":line,"callid":df["callid"][i],"convo":df["convo"][i]}
                        cust.append(line)
                        # speaker.append(spk)
            if len(cust)==0:
                dict2 = [{"callid":df["callid"][i],"convo":df["convo"][i],"l1":"","l2":""}]
                l12.extend(dict2)
                continue
            # doc_embeddings = model.encode(cust)
            similarity_matrix = get_cosine_similarity(cust)
            
            similarity_matrix = np.array(similarity_matrix)

            upper_triangle_indices = np.triu_indices_from(similarity_matrix, k=1)
            upper_triangle_similarities = similarity_matrix[upper_triangle_indices]
            num_pairs = 15

            if num_pairs > upper_triangle_similarities.shape[0]:
                num_pairs = upper_triangle_similarities.shape[0]
            # Get indices of the top similarities
            top_indices = np.argpartition(upper_triangle_similarities, -num_pairs)[-num_pairs:]
            top_indices = top_indices[np.argsort(-upper_triangle_similarities[top_indices])]
        
            # Construct top pairs
            top_pairs = []
            allowed_proximity = 5
            for index in top_indices:
                xi, xj = upper_triangle_indices[0][index], upper_triangle_indices[1][index]
                if abs(xi - xj) <= allowed_proximity:
                    similarity_score = similarity_matrix[xi, xj]
                    top_pairs.append((cust[xi], cust[xj], similarity_score))
                
            # print(f"Top {num_pairs} most similar pairs:")
            p2=[]
            dict_list=[]
            for yi, (line1, line2, score) in enumerate(top_pairs):
        
                if score >= 0.72:
                    c=""
                    found=False
                    for linex in convo.split("\n"):
                        if (line1 in linex) or found==True:
                            c+=linex+"\n"
                            found=True
                        if line2 in linex:
                            found=False
                            break
                        #print(found)
    
                    dict2 = {"convo":convo,"l1":line1,"l2":line2}
                    dict_list.append(dict2)
                
                else:
                    continue

            l11 = dict_list
            for j in range(len(l11)):
                l11[j]["callid"] = df["callid"][i]
                l11[j]["convo"] = df["convo"][i]
                
            if len(l11)!=0:
                l12.extend(l11)
            else:
                dict2 = [{"callid":df["callid"][i],"convo":df["convo"][i],"l1":"","l2":""}]
                l12.extend(dict2)
            
        return l12

    except Exception as e:
        logger.error(f"Repetition - rep_llm_hit: {e}", exc_info=True)
        raise

def rep_llm_hit1(l12,batch_no, model):
    try:
        repetition_prompt_json = json.dumps(repetition_prompt)
        interval = 50
        a=0
        l13 = l12
        j_list=[]
        p2=[]
        p3 = []
        b2=0
        for i in range(len(l12)):
            #for j in range(a,a+interval):
            line1 = l13[i]["l1"]
            line2 = l13[i]["l2"]
            
            if line1 == "":
                l13[i]["out"]="no repeat"
                l13[i]["repeat"]="no"
                b2+=1
                continue
            if line2 == "":
                l13[i]["out"]="no repeat"
                l13[i]["repeat"]="no"
                b2+=1
                continue
            j_list.append(i)
            # p = {'user_prompt':user_prompt.format(line1,line2),'system_prompt':system_prompt}
            # prompt.format(ins,line1,line2)
            p2.append(line1)
            p3.append(line2)
        # # llm_res = generate_llm_resp_batch(p2,params)
        # llm_res = get_llm_responses(p2, params)
        if not p2 or not p3:
            logger.warning("Repetition - rep_llm_hit1: No data to process, skipping DataFrame creation.")
            return pd.DataFrame(l13)
        columns_data = list(zip(p2, p3))
        columns = ["line1", "line2"]
        df_lines = spark.createDataFrame(columns_data, schema=columns)

        llama_request = LLMBatchRequest(model, temperature=0.1, max_tokens=1024)

        # result_df = df_lines.withColumn("result", expr(f"ai_query('databricks-meta-llama-3-1-8b-instruct', request => concat({repetition_prompt_json}, line1, '\nStatement 2 : customer : ', line2), modelParameters => named_struct('temperature', 0.1, 'max_tokens', 1024))"))

        result_df = llama_request.apply_batch(df_lines, f"concat({repetition_prompt_json}, line1, '\nStatement 2 : customer : ', line2)")

        result_df_pd = result_df.toPandas()
        llm_res = result_df_pd["result"].tolist()
        # print(llm_res)
        b=0
        for output in llm_res:
            try:
                matches = re.findall(r"```(.*?)```", output, re.DOTALL)
                json_string = re.findall(r"({.*})", output, re.DOTALL)
                
                json_objects = []
                for match in matches:
                    json_obj = json.loads(match.strip())
                    json_objects.append(json_obj)

                json_objects_2 = []
                for match in json_string:
                    json_obj = json.loads(match.strip())
                    json_objects_2.append(json_obj)

                if len(json_objects)!=0:
                    # print(json_objects[0])
                    output = json_objects[0]  
                elif len(json_objects_2) != 0:
                    # print(json_objects_2[0])
                    output = json_objects_2[0]
                else:
                    # print(output)
                    output = json.loads(output)
                    
            except:
                logger.warning("Repetition - rep_llm_hit1: LLM response JSON decoding issue.")
            
            j2 = j_list[b]
            try:
                o1 = str(output["how_similar_is_situation_of_both_statements"]["answer"])
                o2 = str(output["how_similar_is_objective_of_both_statements"]["answer"])
                
                if "dissimilar" in o1:
                    l13[j2]["repeat"] = "no"
                elif "dissimilar" in o2:
                    l13[j2]["repeat"] = "no"
                elif "very" in o1:
                    l13[j2]["repeat"]="yes"
                elif "very" in o2:
                    l13[j2]["repeat"]="yes"
                else:
                    l13[j2]["repeat"]="no"
                l13[j2]["out"] = output
                b+=1
            except :
                logger.warning("Repetition - rep_llm_hit1: After JSON failiure.")
                b+=1
        return pd.DataFrame(l13)

    except Exception as e:
        logger.error(f"Repetition - rep_llm_hit1: {e}")
        raise
    
def de_dupli_rep(df2,batch_no):
    try:
        df2["multiple_rep"]="no"
        df3 = pd.DataFrame(columns=df2.columns)
        unique_callids = df2['callid'].value_counts() == 1
        print("unique cid : ",len(unique_callids))
        
        # Create DataFrames
        unique_df = df2[df2['callid'].isin(unique_callids[unique_callids].index)]
        print("u1 : ",len(unique_df["callid"].unique()))
        non_unique_df = df2[df2['callid'].isin(unique_callids[~unique_callids].index)]
        print("u2 : ",len(non_unique_df["callid"].unique()))
        n2 = non_unique_df[non_unique_df["repeat"]=="no"]
        n2 = n2.drop_duplicates(subset='callid').reset_index(drop=True)

        non_unique_df = non_unique_df[non_unique_df["repeat"]=="yes"]
        non_unique_df = non_unique_df.sort_values(by='callid').reset_index(drop=True)
        print("u3 : ",len(non_unique_df["callid"].unique()))
        n2 = n2[~n2['callid'].isin(non_unique_df['callid'])]
        n2 = n2.reset_index(drop=True)
        
        if len(non_unique_df)==1:
            non_unique_df = pd.concat([non_unique_df, unique_df], ignore_index=True)
            non_unique_df = pd.concat([non_unique_df, n2], ignore_index=True)
            non_unique_df = non_unique_df.reset_index(drop=True)
            return non_unique_df
        i2=0
        change=True    
        for i in range(len(non_unique_df)-1):
            cid = non_unique_df["callid"][i]
            cid2 = non_unique_df["callid"][i+1]
            #print(cid,cid2,change)
            if cid == cid2:
                if change == True:
                    df3.loc[i2]=non_unique_df.loc[i]
                    df3.loc[i2,"multiple_rep"]="yes"
                    change = False
                    i2+=1
            else:
                
                if change == True:
                    df3.loc[i2]=non_unique_df.loc[i]
                    i2+=1
                
                change = True
        if len(non_unique_df)!=0:
            if non_unique_df["callid"][len(non_unique_df)-1] not in df3["callid"]:
                df3.loc[i2]=non_unique_df.loc[len(non_unique_df)-1]
        df3 = pd.concat([df3, unique_df], ignore_index=True)
        df3 = pd.concat([df3, n2], ignore_index=True)
        df3 = df3.reset_index(drop=True)
        return df3
        
    except Exception as e:
        logger.error(f"Repetition - de_dupli_rep: {e}")
        raise
def extract_chunk(row):   #new
    convo = row['convo']  
    l1 = "CALLER: "+ row['l1']
    l2 = "CALLER: "+ row['l2']
    if pd.isna(convo) or pd.isna(l1) or pd.isna(l2):
        return ""
    try:
        lines = convo.split('\n')
        lines = [line for line in lines if line.strip()]
        lines = merge_consecutive_lines(lines)
        idx1 = next((i for i, line in enumerate(lines) if l1.strip() in line), None)
        idx2 = next((i for i, line in enumerate(lines) if l2.strip() in line), None)
        if idx1 is not None and idx2 is not None and idx1 <= idx2:
        #     return '\n'.join(lines[idx1:idx2+1])
        # elif idx1 is not None:
        #     return lines[idx1]
            start_idx = max(0, idx1 - 2)
            return '\n'.join(lines[start_idx:idx2+1])
        elif idx1 is not None:
            logger.warning(f"extract_chunk- only one line found l1. l2 is missing")
            start_idx = max(0, idx1 - 2)
            return lines[start_idx:start_idx+10]
        else:
            return ""
    except Exception:
        return ""
        logger.error(f"Repetition - extract_chunk: {e}")
        raise

def limit_markdowns_on_proximity(data):   #new
    try:
        data['utterance_count'] = data['conversation_chunks'].apply(lambda x: len(x.split('\n')))
        bod_mask =  data['utterance_count'] >= 12
        data.loc[bod_mask,'repeat'] = 'no'
        return data
    except Exception as e:
        logger.error(f"Repetition - limit_markdowns_on_proximity: {e}")
        return data

def confirm_customer_repetition(data):
    df_rep = data[data['repeat']=='yes']
    if len(df_rep)==0:
        data['result'] = ''
        return data
    df_norep = data[data['repeat']!='yes']
    dfrep_spark = spark.createDataFrame(df_rep)
    llama_request = LLMBatchRequest(model, temperature=0.1, max_tokens=1024)
    result_df = llama_request.apply_batch(dfrep_spark, f"concat('{rep_confirmation_prompt}', l1, '\nline 2 : customer : ', l2, '\ncontext : ', conversation_chunks)")
    result_df = result_df.toPandas()
    for idx, row in result_df.iterrows():
        try:
            res_obj = json.loads(row['result'])
            if res_obj['Repetition'].lower() == 'no':
                result_df.at[idx, 'repeat'] = 'no'
            if res_obj['reason_for_repetition'].lower() != 'agent_asked_for_same_details':
                result_df.at[idx, 'repeat'] = 'no'
        except Exception as e:
            print(f"Error in parsing JSON: {e}")

    return pd.concat([df_norep, result_df])  
def c_rep_main(df,batch_no, model):
    logger.info("Understanding - Repetition is running !!")
    try:
        s_t = time()
        df = df[['callid','channel','transcript']].drop_duplicates()
        df['convo'] = df['channel'] + ": " + df['transcript']
        df = df.groupby('callid')['convo'].apply(concatenate_strings).reset_index()  # pandas series
        df.columns = ['callid', 'convo']  
        indices_to_drop = []

        for index, row in df.iterrows():
            # Your condition here, for example:
            if len(row['convo'].split('\n')) < 15:
                indices_to_drop.append(index)
        df = df.drop(indices_to_drop)
        # Reset index if needed
        df.reset_index(drop=True, inplace=True)
        # print(df.info())
        df99 = df.copy()
        l12 = embed_hit(df,batch_no)
        logger.info(f"Length of l12 : {len(l12)}")
        df2 = rep_llm_hit1(l12,batch_no, model)
        logger.info(f'before de-dupli unique: {len(df2["callid"].unique())}')
        df3 = de_dupli_rep(df2,batch_no)
        # print("originally unique ",len(df99["callid"].unique()))

        df3['conversation_chunks'] = df3.apply(extract_chunk, axis=1)   #new
        df3 = limit_markdowns_on_proximity(df3)
        df3 = confirm_customer_repetition(df3)
        
        logger.info("Understanding - Repetition is completed !!")
        return df3
        
    except Exception as e:
        logger.error(f"c_repetition - c_rep_main: {e}")
        raise