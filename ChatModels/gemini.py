from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv

load_dotenv()

# model object
model = ChatGoogleGenerativeAI(model='gemini-3.8-flash') 

# invoking model
result = model.invoke('What is the capital of himachal pradesh')

print(result.content[0]['text'])