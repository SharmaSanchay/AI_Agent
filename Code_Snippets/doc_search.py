import os
from openai import OpenAI
from dotenv import load_dotenv
import numpy as np

load_dotenv()  # by default looks for .env file


client = OpenAI(base_url=os.getenv("BASE_URL"), api_key=os.getenv("API_KEY"))

EMBED_MODEL = os.getenv("EMBEDDING_MODEL")
KB_FOLDER = os.getenv("KB_FOLDER")


# Load the document from the knowledge base folder
def load_documents(folder=KB_FOLDER):
    documents = []
    for filename in os.listdir(folder):
        if filename.endswith(".txt"):
            with open(os.path.join(folder, filename), "r", encoding="utf-8") as f:
                documents.append({"source": filename, "text": f.read()})
    return documents


# get the embedding for the given text
def get_embedding(text):
    response = client.embeddings.create(model=EMBED_MODEL, input=text)
    return response.data[0].embedding


def cosine_similarity(vec1, vec2):
    """
    This function is used to calculate the cosine similarity between two vectors
    """
    dot_product = np.dot(vec1, vec2)
    norm_vec1 = np.linalg.norm(vec1)
    norm_vec2 = np.linalg.norm(vec2)
    return dot_product / (norm_vec1 * norm_vec2)


# adding the embedding for each documents
def add_embedding(documents):
    for document in documents:
        document["vector"] = get_embedding(document["text"])
    return documents


def search(query, documents, top_k=3):
    query_vector = get_embedding(query)

    result = []

    for document in documents:
        score = cosine_similarity(query_vector, document["vector"])
        result.append(
            {
                "score": float(score),
                "source": document["source"],
                "text": document["text"],
                "size": len(document["text"]),
            }
        )
    result.sort(key=lambda rec:rec["score"],reverse=True)
    return result[:top_k]

#testing method 
def test(user_query,documents,top_k):
    print(f"\n User Query: {user_query}\n")
    results = search(user_query,documents,top_k)

    for r in results:
        print(f"Score : {r['score']:.2f} - {r['source']} - {r['size']} characters")


documents = load_documents()
documents = add_embedding(documents)

print(f"We have embedded {len(documents)} documents")

user_query = "i got a new offer from another company when will i can join that company?"

test(user_query,documents,3)
