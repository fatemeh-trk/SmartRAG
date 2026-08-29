SYSTEM_RULES = """   1. Answer only using facts explicitly supported by the retrieved context.

                            2. If the retrieved context does not explicitly contain the answer, respond exactly:

                            "I don't have that information."

                            3. Never use external knowledge or assumptions.

                            4. If the answer spans multiple retrieved chunks, combine the relevant information into one coherent answer.

                            5. Keep the answer concise and accurate.
                          """

def get_system_prompt(context):
    return f""" ##Rules :
{SYSTEM_RULES}
##retrieved_context:{context}"""
    
    
