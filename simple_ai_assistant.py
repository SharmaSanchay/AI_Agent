# standard library imports
import os
from types import SimpleNamespace

# Third-party imports
from openai import OpenAI
from dotenv import load_dotenv

# Local modules
from Utilties.utlitlies import log
from tools import run_tool, Tools

load_dotenv()  # by default looks for .env file

client = OpenAI(base_url=os.getenv("BASE_URL"), api_key=os.getenv("API_KEY"))


messages = []

# Safety limit: the max number of model rounds for ONE user question
MAX_STEPS = 5

messages.append(
    {
        "role": "system",
        "content": (
            "You are a helpful, humble assistant. "
            "Answer general questions (programming, science, advice) from your own knowledge. "
            "Use a tool ONLY when the user asks for the current time, the weather, or their city. "
            "Keep answers under 100 words. and don't return me count of words in your answer. "
        ),
    }
)

def build_tool_calls_payload(tool_calls):
    payload = []
    for tool_call in tool_calls:
        payload.append(
            {
                "id": tool_call.id,
                "function": {
                    "name": tool_call.function.name,
                    "arguments": tool_call.function.arguments,
                },
            }
        )
    return payload


def stream_chat(messages):
    """
    This function is used to stream the chat response from the model
    """

    stream = client.chat.completions.create(
        model=os.getenv("MODEL"),
        messages=messages,
        tools=Tools,
        stream=True,
    )

    full_response = ""
    calls = {}
    label_printed = False

    # 1. Stream the response from the model
    for chunk in stream:
        if not chunk.choices:
            continue
        delta = chunk.choices[0].delta
        if delta.content:
            if not label_printed:
                print("🤖 Assistant: ", end="", flush=True)
                label_printed = True
            print(delta.content, end="", flush=True)
            full_response += delta.content

        # 2. Check if the model is requesting a tool call
        for tc in delta.tool_calls or []:
            obj = calls.setdefault(
                tc.index,
                SimpleNamespace(
                    id=None, function=SimpleNamespace(name="", arguments="")
                ),
            )

            if tc.id:
                obj.id = tc.id
            if tc.function and tc.function.name:
                obj.function.name += tc.function.name
            if tc.function and tc.function.arguments:
                obj.function.arguments += tc.function.arguments

    #  since we printed the label and response we need to move forward to next line
    if label_printed:
        print()

    # sorting the tool calls based on in the index
    calls_list = []
    for key in sorted(calls):
        calls_list.append(calls[key])
    return full_response, calls_list



# Chat loop starts from here
while True:

    user_input = input("\n👦🏻 Sanchay: ")

    if user_input.lower() in ["exit", "quit", "bye", "goodbye"]:
        break

    messages.append({"role": "user", "content": user_input})

    # Chained Tool Calling
    for step in range(MAX_STEPS):
        full_response, tool_calls = stream_chat(messages)

        if not tool_calls:
            break

        log(f"Step {step + 1}: model requested {len(tool_calls)} tool calls")

        messages.append(
            {
                "role": "assistant",
                "content": full_response,
                "tool_calls": build_tool_calls_payload(tool_calls),
            }
        )

        for tool_call in tool_calls:
            result = run_tool(tool_call)
            messages.append(
                {
                    "role": "tool",
                    "tool_call_id": tool_call.id,
                    "content": str(result),
                }
            )

    else:
        # Ran out of steps: The model never gave the final answer
        fallback = "Sorry, I could not finish this within the step limit."
        messages.append({"role": "assistant", "content": fallback})
        print("🤖 Assistant: " + fallback)
        continue
