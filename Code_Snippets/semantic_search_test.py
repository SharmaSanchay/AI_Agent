import os 
from openai import OpenAI
from dotenv import load_dotenv
import numpy as np

load_dotenv()  # by default looks for .env file


client = OpenAI(base_url=os.getenv("BASE_URL"), api_key=os.getenv("API_KEY"))

def embedding(text):
    """
    This function is used to get the embedding of the text
    """
    response = client.embeddings.create(
        model=os.getenv("EMBEDDING_MODEL"),
        input=text,
    )
    return response.data[0].embedding

def cosine_similarity(vec1, vec2):
    """
    This function is used to calculate the cosine similarity between two vectors
    """
    dot_product = np.dot(vec1, vec2)
    norm_vec1 = np.linalg.norm(vec1)
    norm_vec2 = np.linalg.norm(vec2)
    return dot_product / (norm_vec1 * norm_vec2)

text1 = "I love programming in Python."
text2 = "Python is my favorite programming language."
text3 = "I enjoy hiking and outdoor activities."


embedding1 = embedding(text1)
embedding2 = embedding(text2)
embedding3 = embedding(text3)

similarity1_2 = cosine_similarity(embedding1, embedding2)
similarity1_3 = cosine_similarity(embedding1, embedding3)

print(f"Cosine similarity between text1 and text2: {similarity1_2:.2f}")
print(f"Cosine similarity between text1 and text3: {similarity1_3:.2f}")