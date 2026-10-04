from langchain_huggingface import HuggingFaceEndpointEmbeddings
from dotenv import load_dotenv
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

load_dotenv()

embeddings = HuggingFaceEndpointEmbeddings(
    model="sentence-transformers/all-MiniLM-L6-v2"
)

doc = [
    "Virat Kohli is an Indian cricketer known for his aggressive batting and leadership.",
    "MS Dhoni is a former Indian captain famous for his calm demeanor and finishing skills.",
    "Sachin Tendulkar, also known as the 'God of Cricket', holds many batting records.",
    "Rohit Sharma is known for his elegant batting and record-breaking double centuries.",
    "Jasprit Bumrah is an Indian fast bowler known for his unorthodox action and yorkers."
]

query = "Tell me about Jasprit Bumrah"

embed_doc = embeddings.embed_documents(doc)
embed_query = embeddings.embed_query(query)

#cosine_similarity() me dono 2d vector pass hote hai 
scores = cosine_similarity([embed_query],embed_doc)[0]

index,score = sorted(list(enumerate(scores)),key=lambda x:x[1])[-1]

print(query)
print(doc[index])
print("Similarity score is",score)
