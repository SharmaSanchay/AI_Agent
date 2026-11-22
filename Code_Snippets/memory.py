import os
import json
from datetime import datetime
from dotenv import load_dotenv

load_dotenv()

CHAT_FOLDER = os.getenv("CHAT_FOLDER")

def save_chat(path,messages):
    os.makedirs(CHAT_FOLDER, exist_ok=True)

    if path is None:
        filename = datetime.now().strftime("%Y%m%d%H%M%S") + ".json"
        path = os.path.join(CHAT_FOLDER, filename)

        data = {
            "title":messages[0]['content'][:40],
            "updated": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "messages": messages
        }

        tmp_path = path + '.tmp'
        with open(tmp_path, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=4)
        os.replace(tmp_path, path)

        return path 

def load_chat(path):
    with open(path,'r',encoding='utf-8') as f:
        data = json.load(f)
    return data

def list_chats():
    if not os.path.exists(CHAT_FOLDER):
        return []
    chats = []
    for chat_file_name in os.listdir(CHAT_FOLDER):
        if chat_file_name.endswith(".json"):
            path = os.path.join(CHAT_FOLDER, chat_file_name)
            data = load_chat(path)
            if data is None:
                print(f"Failed to load chat from {path}. Skipping.")
                continue
            chats.append(
                {
                    "path": path,
                    "title": data.get("title", "Untitled Chat"),
                    "updated": data.get("updated", ""),
                    "count": len(data["messages"]),
                }
            )
    chats.sort(key=lambda x:x['updated'],reverse=True)
    return chats
    