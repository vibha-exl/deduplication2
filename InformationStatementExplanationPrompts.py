system_prompt = """You are assistant who helps to audit the given conversation between customer/caller and agent.
1. Answer the given questions based on transcript in Yes/No in JSON format.
2. Validate the information like credit limit etc. if mentioned in Transcript against the data appended from Database. The validation shoud NOT be strict and transcription errors should be kept in mind.
3. Return only in valid JSON following the given schema.
4. When validating information of the Trascript and Database, if that information is not mentioned in transcript then return N.A. 
IMPORTANT: Be linient in validating information between Transcript and Database. 
IMPORTANT: Don't give any explanation only JSON output.
IMPORTANT: Don't put anything else apart from proper JSON in output."""

user_prompt_account_balance = """{"Does the call involve Account Balance Request?":"Yes/No",
                          "Did the agent give information about current balance of customer/ billed balance on account to the customer?": "Yes/No",
                          "Did the agent give all the asked details correctly as per database?": "Yes/No",
                          "Did the agent give information about unbilled goods amount/good on approval to the customer?": "Yes/No",
                          "Did the agent give information about total amount/total commitment amount to the customer?": "Yes/No",
                          "Did the agent give information about the credit limit to the customer?": "Yes/No",
                          "Did the agent give information about the remaining credit to the customer?": "Yes/No",
                          "Did the agent give information about required payment amount/minimum monthly payment to the customer?": "Yes/No",
                          "Did the agent give information about when the payment is due by to the customer?": "Yes/No",
						  "Was the credit limit provided by the agent correct as per database?": "Yes/No",
						  "Was the Last Statement date provided by the agent correct as per database?": "Yes/No",
						  "Was the Last Statement number provided by the agent correct as per database?": "Yes/No",
						  "Was the Next Statement Date provided by the agent correct as per database?": "Yes/No",
						  "Was the Next Statement Number provided by the agent correct as per database?": "Yes/No",
						  "Was the Last Payment Amount provided by the agent correct as per database?": "Yes/No",
						  "Was the Last Payment Date provided by the agent correct as per database or equal Call date?": "Yes/No",
                        "Was the Last Order Date provided by the agent correct as per database or equal Call date?": "Yes/No",
                         "Was the Required Payment Due Date provided by the agent correct as per database?": "Yes/No",
                         "Did the customer/caller explicitly ask for account balance?": "Yes/No",
                         "Did the customer/caller wanted to know their balance?": "Yes/No",
                         "Did the customer/caller ask about the statement date clearly/explicitly?" : "Yes/No",
                         "Did the caller wanted to know about the statemet date?" :"Yes/No",
                         "Does the agent confidently/clearly mention the required details asked by caller?": "Yes/No",
                         "Did the agent explanation resolve the doubts for caller?" : "Yes/No",
                         "Is the caller confused/did not understand/not agree on the balance explanation?": "Yes/No",

                          "Evidence": "Summarized reason (very brief) in support of the answers. DO NOT USE QUOTES IN THIS ANSWER"
} """


user_prompt_place_an_order = """ {
"Is the call about Placing an order/involve delivery charge?": "Yes/No",
"Did the caller ask what is the delivery charge?": "Yes/No",
"Did the agent inform the Delivery charge Parcel Shop as 3.50?": "Yes/No",
"Did the agent inform the Delivery charge for Home delivery as 4.95?": "Yes/No",
"Did the agent inform that delivery status tracking email will be sent for home delivery?": "Yes/No",
"Did the agent inform that email/text message will be sent when item can be collected for store delivery?": "Yes/No",
"Does the call involve collection of order from store?": "Yes/No",
"Did the agent inform the customer to carry their ID with them when collecting order from the store?": "Yes/No",
"Is the caller making a payment in the call?":"Yes/No",
"Was the payment successful?":"Yes/No",
"Did advisor mention that a confirmation of payment will be sent to registered email id/address?":"Yes/No",
"Was it mentioned in the transcript explicitly that payment was successful?": "Yes/No",
"Did the caller specifically ask the agent the charge on home delivery?": "Yes/No",
"Did the caller specifically ask the agent the charge on parcel shop delivery?":"Yes/No",
"Did the transcript mention refund of charges?": "Yes/No",
"Does the call explicitly involve home delivery of products?": "Yes/No",
"Was a store pickup booked during the call?": "Yes/No",
"Evidence": "Summarized reason (very brief) in support of the answers. DO NOT USE QUOTES IN THIS ANSWER"
}
"""

user_prompt_arrange_a_collection = """ {
"Is the call about arranging an order collection?": "Yes/No",
"Did the caller ask what is the collection charge?": "Yes/No",
"Did the agent inform the Collection charge Parcel Shop as 2.50?": "Yes/No",
"Did the agent inform the Collection charge for Home delivery as 2.50?": "Yes/No",
"Did the agent inform there is no Collection charge for store collection?": "Yes/No",
"Did the agent specifically inform the caller that return charge is applicable for collection arranged?":"Yes/No",
"Was it a case of faulty item or wrong goods sent?":"Yes/No",
"Did the caller specifically ask the agent the charge on home delivery collection?": "Yes/No",
"Did the caller specifically ask the agent the charge on parcel shop collection?":"Yes/No",
"Did the transcript mention refund of charges?": "Yes/No",
"Did the agent say on transcript the collection charges will be refunded?":"Yes/No",
"Was the collection charges will be refunded promised in the trancript?":"Yes/No",
"Did the agent say email confirmation will be sent for refund of charges?": "Yes/No",
"Evidence": "Summarized reason (very brief) in support of the answers. DO NOT USE QUOTES IN THIS ANSWER"
}
"""


user_prompt_account_information = """{

"Does the conversation involve caller having more than one NEXT account?": "Yes/No",
"Did the agent say that the customer can have a cash as well as credit account?": "Yes/No",
"Did the agent say that the customer can have two credit accounts with NEXT?": "Yes/No",
"Does the conversation involve caller having more than one account?": "Yes/No",
"Evidence Account Info": "Summarized reason (very brief) in support of the answers. DO NOT USE QUOTES IN THIS ANSWER"
} """