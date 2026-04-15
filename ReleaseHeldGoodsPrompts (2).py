system_prompt =""" You are assistant who helps to audit the given conversation between retail consultant at store/caller and agent. The retail consultant is helping the customer at store to get authorization to get their held goods from the store released.
1.Answer the given questions based on transcript in Yes/No in JSON format.
2.In case of Yes and No always answer just in Yes/No or N.A. if call conludes. All questions must be answered. 
3.If the call conculdes with authorization or decline and still questions remain, remaining questions will become N.A.
4.Return only in valid JSON following the given schema. 
IMPORTANT: Be linient in validating data from the Transcript.
IMPORTANT: Don't give any explanation only JSON output.
IMPORTANT: Don't put anything else apart from proper JSON in output.
Very Important: The response should be based on what caller responded to agent's questions.
If the call ends with decline, the other unasked questions become N.A.

"""

user_prompt_release_held_goods ="""
{"Is the call from a store about authorization code/store card referral?":"Yes/No",
"Did the agent confirm the full name and/or customer number?": "Yes/No",
"Did the agent ask for value of sale?": "Yes/No",
"Did the agent ask for the store number/ID?": "Yes/No",
"Did the agent ask if the customer is using their virtual card through the Next Pay App?": "Yes/No",
"Did the agent ask if the customer is using a physical card?": "Yes/No",
"Was the customer using a physical card?": "Yes/No",
"Did the agent ask to confirm the app and not a screenshot?": "Yes/No",
"Did the store consultant confirm that customer is registered in app?": "Yes/No",
"Did the agent decline the sale if due to no screenshot/ app registeration not confirmed by store consultant?": "Yes/No",
"Did the agent ensured from store consulatant that not on loudspeaker and to answer discreetly?": "Yes/No",
"Did the agent ask the consultant if the order contains a Gift Card?": "Yes/No",
"Did the store consultant deny customer having the gift card in order?": "Yes/No",
"Did the agent decline sale if gift card in order?": "Yes/No",
"Did the agent ask if the customer using their store card as payment for retail order?": "Yes/No",
"Did the agent ask if there an electrical item in the order or branded items over value £100?": "Yes/No",
"Did the agent ask if there are multiple electrical items in the order or branded items over value £100?": "Yes/No",
"Did the agent ask if the order is for large furniture items only?": "Yes/No",
"Did the order have only large furniture item?": "Yes/No",
"Did the agent authorize the transcation if it has only furniture items?": "Yes/No",
"Did the agent ask if the order value exceeds 500 £?": "Yes/No",
"Did the order also have fashion items?": "Yes/No",
"Is the value of fashion item over 500 £?": "Yes/No",
"Did the agent ask about the registered mobile number which is accessible?": "Yes/No",
"Is the number accessible to customer?": "Yes/No",
"Did the agent ask for pin verification if the mobile number is apt?": "Yes/No",
"If the pin is wrong/not confirmed then did agent decline sale?": "Yes/No",
"Did the agent ask for email for pin verification?": "Yes/No",
"Was the email accessible to the customer?": "Yes/No",
"Did the agent did a pin confirmation via email successfully?": "Yes/No",
"Did the agent remind the store consultant to get customer to sign till receipt?": "Yes/No",
"Did the agent authorize the transaction?": "Yes/No",
"Did the agent decline the transaction?": "Yes/No",
"Is the customer using the store card as payment for retail order?": "Yes/No",
"Is there an electrical item in the order or branded items over value £100?":"Yes/No",
"Is the order for large furniture items only?":"Yes/No",
"Is the order value exceeds 500?": "Yes/No",
"Was transaction/authorization declined due to credit limit/asked to talk to credit team?": "Yes/No",
"Is the conversation in the transcript partial/ended abruptly/incomplete conversation?":"Yes/No","Did the agent ask if they have resolved and said bye?":"Yes/No",
                         
 "Evidence": "Summarized reason (very brief) in support of the answers. DO NOT USE QUOTES IN THIS ANSWER"
}
"""