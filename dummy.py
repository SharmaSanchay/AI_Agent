import os
from openai import OpenAI
from dotenv import load_dotenv
from tools import Tools , execute_tool
from Utilties.utlitlies import log

load_dotenv()

client = OpenAI(base_url=os.getenv("BASE_URL"), api_key=os.getenv("API_KEY"))
MAX_STEPS = int(os.getenv("MAX_TOOLS_CALL_COUNT", 3))

#  this array is the context of the our model
messages = [
    {
        "role": "system",
        "content": "You are a helpful assistant. which gives quick and short answers to the user queries. in 10 words",
    }
]



while True:
    # step 1:  taking the user query
    query = input("👨‍💻 Ask a question (or type 'exit' to quit): ")
    if query.lower() in ["exit", "quit", "goodbye", "bye"]:
        break
    messages.append({"role": "user", "content": query})

    #  step 2: calling the model to get the response and calling the tools if neded
    for step in range(MAX_STEPS):
        tool_count = 0
        #  initate the chat with llm
        response = client.chat.completions.create(
            model=os.getenv("MODEL"), messages=messages, tools=Tools
        )
        msg = response.choices[0].message
        messages.append(msg)

        #  now check if the tool call is needed or not if not needed break from there
        if not msg.tool_calls:
            break

        log(f"Step {step + 1 } : model requested {len(msg.tool_calls)} tool calls.")

        #  calling tools
        for tool_call in msg.tool_calls:
            # check if the tool call count has reached the maximum limit
            if tool_count >= MAX_STEPS:
                result = "Maximum tool calls reached. Stopping further tool calls."
                log(result)
                break
                 
            result = execute_tool(tool_call)
            tool_count += 1
            messages.append(
                {
                    "role": "tool",
                    "tool_call_id": tool_call.id,
                    "content": result,
                }
            )
    else:
        #Ran out of steps : the model never gave a final answer, so we ask it for one now.         
        fallback = "The model did not provide a final answer within the allowed steps. Please provide a final response."
        messages.append({"role": "assistant", "content": fallback})
        print("🤖 Assistant:", fallback)
        continue 
    print("🤖 Assistant:", msg.content)
