system_prompt =""" You are assistant who helps to audit the given conversation between customer/caller and agent.
1. Answer the given questions based on transcript in Yes/No in JSON format.
2. Validate the account changes done in Transcript against the data appended from Database. The validation shoud NOT be strict and transcription errors should be kept in mind.
3. Return only in valid JSON following the given schema. 
IMPORTANT: Be linient in validating data changes between Transcript and Database.
IMPORTANT: Don't give any explanation only JSON output.
IMPORTANT: Don't put anything else apart from proper JSON in output.
VERY IMPORTANT: For questions like Does the conversation involve address change and so on, the answer should be YES ONLY when most of the conversation involves address change / the address change proccess if followed in the conversation.
VERY IMPORTANT: The conversation is about address change only if the change takes place during conversation.
Similarly for name change, phone number change, email change, account closure etc - entire conversation should have that proccess. 
IMPORTANT: ALL answers should be Yes or No , N.A. - not allowed. 
"""
user_prompt_name_change = """
{"Does the conversation involve name change?":"Yes/No",
"Did the name change happen during conversation/trancript?":"Yes/No",
 "Was the name change validation correct?:"Yes/No",
 "Did the name in transcript even slightly match the database?":"Yes/No",
 "Is the conversation in the transcript partial/ended abruptly/incomplete conversation?":"Yes/No",
 "Did the agent ask if they have resolved and said bye?":"Yes/No",
 "Evidence": "Summarized reason (very brief) in support of the answers. DO NOT USE QUOTES IN THIS ANSWER"
}
"""
user_prompt_name_change_cash_account = """
{"Does the conversation involve name change?":"Yes/No",
 "Does name change involve only title changes (like miss Mrs Mr)?":"Yes/No",
 "Did the name change happen during conversation/trancript?":"Yes/No",
 "If full name change/first name change, did agent ask to close account?":"Yes/No",
 "Was the name change validation correct?:"Yes/No",
 "Did the name in transcript even slightly match the database?":"Yes/No",
 "Is the conversation in the transcript partial/ended abruptly/incomplete conversation?":"Yes/No",
  "Did the agent ask if they have resolved and said bye?":"Yes/No",
 "Evidence": "Summarized reason (very brief) in support of the answers. DO NOT USE QUOTES IN THIS ANSWER"
}
"""
user_prompt_telephone_change = """
{"Does the conversation involve telephone change?":"Yes/No",
"Did the phone/cell/contact/mobile number change happen during conversation/trancript?": "Yes/No",
 "Was the telephone change validation correct?:"Yes/No",
 "Did the phone/cell/contact/mobile number in transcript even slightly match the database?":"Yes/No",
 "Does the conversation involve no change due to 24hrs security?": "Yes/No",
 "Is the conversation in the transcript partial/ended abruptly/incomplete conversation?":"Yes/No",
  "Did the agent ask if they have resolved and said bye?":"Yes/No",
 "Evidence": "Summarized reason (very brief) in support of the answers. DO NOT USE QUOTES IN THIS ANSWER"
}
"""
user_prompt_address_change = """
{"Does the conversation involve address change?":"Yes/No",
"Did the agent ask if caller wants to place an order today?": "Yes/No",
"Was there any conversation regarding placing orders/placed orders?": "Yes/No",
"Was there any discussion about order being placed/re-ordered/cancelled/processed?": "Yes/No",
"Was there any discussion about any orders?": "Yes/No",
"Does the caller wants to place an order?": "Yes/No",
"Did the agent ask caller to allow until tomorrow before placing next order?": "Yes/No",
"Did the agent ask if caller has access to the mobile/phone number?": "Yes/No",
"Did the conversation involve discussion over caller's registered telephone number/mobile number/any phone?":"Yes/No",
"Did the agent ask any additional security questions or digits of card?": "Yes/No",
"Was any PIN discussed in the conversation?": "Yes/No",
"Did the pin verification through phone fail?": "Yes/No",
"If the pin verification fail, were the security questions asked?": "Yes/No",
"Did the agent ask the caller if contact number needs to be changed?": "Yes/No",
"Did the agent ask the caller if email address needs to be changed?": "Yes/No",
"Did the agent confirm address change and that a letter will be sent to new address?": "Yes/No",
"Was the address change validation correct?:"Yes/No",
"In the conversation is the address change proccess actually followed/change done during call?":"Yes/No",
"Is the conversation in the transcript partial/ended abruptly/incomplete conversation?":"Yes/No",
"Did the agent mention that the pin verification fail?", "Yes/No",
 "Did the agent ask if they have resolved and said bye?":"Yes/No",
"Evidence": "Summarized reason (very brief) in support of the answers. DO NOT USE QUOTES IN THIS ANSWER"
}
"""
user_prompt_email_change = """
{"Does the conversation involve email change?":"Yes/No",
"Does conversation about any previous phone change so additional security for email?": "Yes/No"
"Did the agent ask about the phone being accessible for pin verification?": "Yes/No",
"Did the pin verification through phone happen for email change?": "Yes/No",
"Did the conversation have anything about pin verification failing/did pin verification fail?":"Yes/No",
"Did the email change happen during the conversation/transcipt?":"Yes/No",
"If the pin verification fail, did the agent ask security questions?": "Yes/No",
"After pin verification, did agent confirm email change?": "Yes/No",
"Was the email change validation correct?:"Yes/No",
"Is the conversation in the transcript partial/ended abruptly/incomplete conversation?":"Yes/No",
 "Did the agent ask if they have resolved and said bye?":"Yes/No",
"Evidence": "Summarized reason (very brief) in support of the answers. DO NOT USE QUOTES IN THIS ANSWER"
} 
"""
user_prompt_account_closure = """
{
    "Does part of conversation have account closure?":"Yes/No",
    "Did the agent ask about pending orders?": "Yes/No",
    "Did the caller have any pending orders?" "Yes/No",
    "If the order is pending, did the agent ask the caller to call back after order delivery?", "Yes/No",
    "Did the agent ask about any refund or items to return?": "Yes/No",
    "Did the caller have pending returns/refunds?": "Yes/No",
    "If the returns/refund is pending, did the agent ask the caller to call back after?", "Yes/No",
    "Did the agent ask for the reason for closure?": "Yes/No",
    "Did agent ask for the account closure confirmation?”: "Yes/No",
    "Was the account closure done?": "Yes/No",
    "Is the conversation in the transcript partial/ended abruptly/incomplete conversation?":"Yes/No",
     "Did the agent ask if they have resolved and said bye?":"Yes/No",
    "Evidence": "Summarized reason (very brief) in support of the answers. DO NOT USE QUOTES IN THIS ANSWER"
}

"""