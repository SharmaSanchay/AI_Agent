import os
from openai import OpenAI
import json
from dotenv import load_dotenv
import numpy as np

load_dotenv()  # by default looks for .env file


client = OpenAI(base_url=os.getenv("BASE_URL"), api_key=os.getenv("API_KEY"))

EMBED_MODEL = os.getenv("EMBEDDING_MODEL")
KB_FOLDER = os.getenv("KB_FOLDER")
VECTOR_STORE_FILE = os.getenv("VECTOR_STORE_FILE")


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


# add the embedding for each chunk
def add_embeddings(chunks):
    for chunk in chunks:
        chunk["vector"] = get_embedding(chunk["text"])
    return chunks


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
                "chunk_id": document["chunk_id"],
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
        print(
            f"Score : {r['score']:.2f} - "
            f"{r['source']} (chunk {r['chunk_id']}) - {r['size']} characters"
        )

# reading the data of existing vector store file
def load_vector_store():
    with open(VECTOR_STORE_FILE, "r", encoding="utf-8") as f:
        return json.load(f)


#  spliting the documents into chunks based on the double new line
def chunk_document(document):
    chunks = []

    chunk_id = 0

    for part in document["text"].split("\n\n"):
        if part.strip():
            chunks.append(
                {
                    "source": document["source"],
                    "chunk_id": chunk_id,
                    "text": part.strip(),
                }
            )
            chunk_id += 1
    return chunks

#  create the chunks
def create_chunks():
    documents = load_documents()
    chunks = []

    for document in documents:
        chunks.extend(chunk_document(document))
    return chunks

#  saving the embedding of the knowledge base to the vector store file
def save_vector_store(chunks):
    with open(VECTOR_STORE_FILE, "w", encoding="utf-8") as f:
        json.dump(chunks, f, ensure_ascii=False, indent=4)



# check if the embeeding of knowledge base is already created or not, if not then create it
if os.path.exists(VECTOR_STORE_FILE):
    # get the existing emebedding of the knowledge base
    chunks = load_vector_store()
    print(f"Loaded {len(chunks)} chunks from the vector store file {VECTOR_STORE_FILE}")
else:
    # create the embedding of the knowledge base and save it to the vector store file
    chunks = create_chunks()
    chunks = add_embeddings(chunks)
    save_vector_store(chunks)
    print(f"Created and embedded {len(chunks)} chunks and saved to the vector store file {VECTOR_STORE_FILE}")



print(f"We have embedded {len(chunks)} documents")
# now we can test the search function with a user query
user_query = "i got a new offer from another company when will i can join that company?"
test(user_query,chunks,3)
