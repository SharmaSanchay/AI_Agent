from datetime import datetime
import requests
import json
from Code_Snippets.rag import search
from Utilties.utlitlies import log


def get_current_time():
    print("get currrent time tool is called")
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def get_current_weather_by_city(city: str = ""):
    print("weather tool is called with city:", city)
    url = f"https://wttr.in?format=3"
    if city:
        url = f"https://wttr.in/{city}?format=3"
    try:
        response = requests.get(url)
        if response.status_code == 200:
            return response
        else:
            return f"Error: Unable to fetch weather data. Status code: {response.status_code}"
    except requests.exceptions.RequestException as e:
        return f"Error: {e}"


def get_current_city():
    return "Pilani"


def search_knowledge_base(query):
    results = search(query)
    log(f"📚 Knowledge base search: {query}")
    for r in results:
        log(f"   {r['score']:.2f} - {r['source']} #{r['chunk_id']}")

    blocks = []
    for r in results:
        sourceFileName = r['source']
        chunkId        = r['chunk_id']
        chunkText      = r['text']

        header = f"[{sourceFileName} #{chunkId}]"
        block  = f"{header}\n{chunkText}"
        blocks.append(block)

    return "\n\n".join(blocks)




Tools = [
    {
        "type": "function",
        "function": {
            "name": "get_current_time",
            "description": "this is used to get the current data time as formatted string",
            "parameters": {"type": "object", "properties": {}},
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_current_weather_by_city",
            "description": "this is used to get the current weather data for a specific city",
            "parameters": {
                "type": "object",
                "properties": {
                    "city": {"type": "string", "description": "the name of the city"}
                },
                "required": ["city"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_current_city",
            "description": "this function is used get the current city of the user so whenever user says my home location , my current location etc then use this function to get actually city of the user ",
        },
        "parameters": {},
    },
    {
        "type": "function",
        "function": {
            "name": "search_knowledge_base",
            "description": "Searches the Zorvex Technologies company knowledge base: leave, medical, HR, hiring, tech, incident "
            "reports and travel expense policies. Use it for any question about company policies or documents. "
            "Returns the most relevant passages with their source.",
        },
        "parameters": {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "The user's question, rewritten as a short, self-contained search query.",
                }
            },
            "required": ["query"],
        },
    },
]

FUNCTION_NAME_TO_FUNCTION_MAPPING = {
    "get_current_time": get_current_time,
    "get_current_weather_by_city": get_current_weather_by_city,
    "get_current_city": get_current_city,
    "search_knowledge_base": search_knowledge_base,
}


def run_tool(tool_call):
    func_name = tool_call.function.name
    result = None
    try:
        func_arg = json.loads(tool_call.function.arguments or "{}")
        log(f" ⚙️ Executing tool: {func_name} with arguments: {func_arg}")

        fn = FUNCTION_NAME_TO_FUNCTION_MAPPING.get(func_name)

        if fn is None:
            raise ValueError(f"Unknown tool: {func_name}")

        result = fn(**func_arg)

        if isinstance(result, str):
            return result

        result = json.dumps(result, default=str, ensure_ascii=False)

    except Exception as e:
        result = f"Tool error ({func_name}): {str(e)}"
    log(f" ⚙️ Tool {func_name} executed successfully with result: {result}")
    return result
