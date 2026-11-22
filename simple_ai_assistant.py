# standard library imports
import os
from types import SimpleNamespace

# Third-party imports
from openai import OpenAI
from dotenv import load_dotenv

# Local modules
from Code_Snippets.memory import list_chats, load_chat,save_chat
from Utilties.utlitlies import log
from tools import run_tool, Tools

load_dotenv()  # by default looks for .env file

client = OpenAI(base_url=os.getenv("BASE_URL"), api_key=os.getenv("API_KEY"))


messages = []

# Safety limit: the max number of model rounds for ONE user question
MAX_STEPS = 5

# SYSTEM_PROMPT = {
#         "role": "system",
#         "content": "You are a helpful, humble assistant for Zorvex Technologies employees. "
#             "Answer general questions (programming, science, advice) from your own knowledge. "
#             "Use get_current_time, get_current_weather_by_city and get_current_city only for the current time, the weather, or the user's city. "
#             "For ANY question about Zorvex policies, benefits, leave, medical cover, HR, hiring, tech processes, incident reports or travel expenses, "
#             "you MUST first call search_knowledge_base. "
#             "Answer company questions ONLY from the text that search_knowledge_base returns. Do not use outside knowledge for company facts. "
#             "After the answer, cite the source like (leave_policy.txt #3). "
#             "If the returned text does not contain the answer, say exactly: I could not find this in the knowledge base. "
#             "Keep answers under 100 words."
#     }

SYSTEM_PROMPT = {
    "role": "system",
    "content": "You are a helpful, humble assistant"
}

messages.append(SYSTEM_PROMPT)


print('this the messages', messages)
chat_file_path = None  # will be set when the chat is saved

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

def show_chats(chats):
    chat_number = 1
    for chat in chats: 
        print(f"{chat_number}. {chat['title']} (Last updated: {chat['updated']}, Messages: {chat['count']})")
        chat_number += 1

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

existing_chats = list_chats()

if existing_chats:
    last_chat = existing_chats[0]
    answer = input(f"Do you want to continue the last chat '{last_chat['title']}'? (y/n): ")
    if answer.lower() == 'y':
        log('Continuing the last chat')
        chat_data = load_chat(last_chat['path'])
        if chat_data is not None:
            messages = [SYSTEM_PROMPT] + chat_data['messages']
            chat_file_path = last_chat['path']
            print(f" {chat_file_path} Continuing chat '{last_chat['title']}' with {len(messages)} messages.")

# Chat loop starts from here
while True:

    user_input = input("\n👦🏻 Sanchay: ")

    if user_input.lower() in ["exit", "quit", "bye", "goodbye"]:
        break

    command = user_input.strip().lower()

    if command == '/new':
        messages = [SYSTEM_PROMPT]
        chat_file_path = None
        print("Started a new chat. The old one is still saved in the chat history.")
        continue

    if command == '/chats':
        chats = list_chats()
        if chats: 
            show_chats(chats)
        else:
            print("No previous chats found.")
        continue

    if command == '/resume':
        chats = list_chats()
        if not chats:
            print("No previous chats found.")
            continue

        show_chats(chats)
        chat_number = input("Enter the number of the chat you want to resume: ")
        chat_number = chat_number.strip()
        if not chat_number:
            print("No input provided. Please enter a valid number.")
            continue
        if not chat_number.isdigit() or not (1<=int(chat_number)<=len(chats)):
            print("Invalid choice. Please enter a valid number.")
            continue
        picked_chat = chats[int(chat_number)-1]
        data = load_chat(picked_chat['path'])
        if data is None:
            print(f"Failed to load chat from {picked_chat['path']}.")
            continue
        messages = [SYSTEM_PROMPT] + data['messages']
        chat_file_path = picked_chat['path']
        print(f"Resuming chat '{picked_chat['title']}' with {len(messages)} messages.")
        continue

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
        chat_file_path = save_chat(chat_file_path, messages[1:])
        print("🤖 Assistant: " + fallback)
        continue
    chat_file_path = save_chat(chat_file_path, messages[1:])
