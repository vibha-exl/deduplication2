system_prompt =""" You are an efficient assistant/auditor that extracts the customer's query/problem from a conversation between agent and customer/caller. Answer the given related questions and return the answer in JSON format. PLEASE RETURN ONLY IN JSON FORMAT; Don't write anything else.
VERY IMPORTANT: DO NOT WRITE ANYTHING ELSE IN YOUR RESPONSE, JUST THE JSON FORMAT.
DO NOT WRITE: Here is the response in JSON format etc , answer only in given JSON format.
"""

user_prompt_transfers_correctly ="""{

"Summarized_reason_transfer":"You have to detect the query/problem of the customer and summarize in not more than 20 words from the given Transcript. 
IMPORTANT: Give the summarized query in not more than 20 words and nothing else. The summary should  be from the transcript only.NO ASSUMPTIONS.If no reason can be infered from the transcript, then write REASON NOT DEFINED.
IMPORTANT: Summary should have(if applicable, if not present in transcript, don't make assumptions)-
 1)product/service 
 2)customer issue 
 3)customer intent
 4)special account type 
 5)reason for transfer by agent description. DO NOT mention/assume the team it is transferred in summary. DO NOT mention  team/department of transfer in the summary.
 All the above should be in format of short summary. 
IMPORTANT-
1)DON'T WRITE customer query is  or anything, just the summarized query in 20 words only. 
2)Summary should strictly include the caller QUERY and REASON of transfer.
4)Exculde transfer details please keep only reason and query.
3)The Summary should help understand why the call is being transferred to a different department.",

"Are there multiple intents/reasons of caller in the transcript/which need different transfers?":"Yes/No",

"Did the agent directly transferred the call after listening to caller query/after customer identification?": "Yes/No",

"Did the caller ask the agent to transfer the call to a particular department/queue/person?": "Yes/No",

"Does the transcript seem abruptly incomplete before transfer?": "Yes/No",

"Was there any injury or harm to the caller due to a product?": "Yes/No",

"Is the caller calling about goods not recieved?": "Yes/No",

"Did the transcript talk about an international order?": "Yes/No",

"Is there conversation about branded items?": "Yes/No",

"Is the caller facing financial difficulties/under stress?": "Yes/No",


"CONTEXT": "Extract the EXACT UTTERANCE WITH SPEAKER VERBATIM inculding whoes utterance is extracted 
Should have QUERY OF THE CALLER.
PLEASE INCULDE THE SPEAKER of the EXTRACTED UTTERANCE.
There could be multiple utterances to support summary but only relevant and very brief.
(relevant for reason for transfer) from the transcript. 
SHOULD BE FROM TRANSCRIPT ONLY to support the given summarized reason for transfer. 
MAXIMUM 40 words.
Reason of transfer should be there.
Where it is being transferred need NOT be incukded
PLEASE DO NOT USE QUOTES IN THIS ANSWER"

}
"""

