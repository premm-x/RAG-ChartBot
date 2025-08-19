
import os
from data import apikey
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from langchain.memory import ConversationBufferMemory

os.environ["GOOGLE_API_KEY"] = apikey
llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash")

memory = ConversationBufferMemory(memory_key="memo")

prompt = PromptTemplate(
    input_variables=["memo","human_input"],

    template="""
        You are a helpful chatbot. 
        The conversation so far: {memo}

        Human: {human_input}
        Chatbot:"""
)

chain = LLMChain(llm = llm, prompt = prompt, memory = memory)


print(chain.run("Hi, my name is Tony."))
print(chain.run("What is my name?"))








