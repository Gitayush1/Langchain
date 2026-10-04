from langchain_huggingface import ChatHuggingFace,HuggingFaceEndpoint
from dotenv import load_dotenv
from langchain_core.messages import SystemMessage,HumanMessage,AIMessage

load_dotenv()

llm = HuggingFaceEndpoint(
    repo_id = "meta-llama/Llama-3.1-8B-Instruct",
    task = "text-generation"
)

model = ChatHuggingFace(llm=llm)

chatHistory = [
    SystemMessage(content="You are a helpful AI Assistant")
]

while True:
    user_input = input("You: ")
    chatHistory.append(HumanMessage(content = user_input))
    if(user_input=="exit"):
        break
    result = model.invoke(chatHistory)
    chatHistory.append(AIMessage(content = result.content))
    print("AI: ",result.content)
print(chatHistory)
