# Install dependencies
# pip install langchain langchain-community langchain-google-genai faiss-cpu pypdf

import os
from data import apikey
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain_community.vectorstores import FAISS
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.prompts import ChatPromptTemplate

# ----------- Set your API key -----------
os.environ["GOOGLE_API_KEY"] = apikey

# ----------- Load PDF -----------
loader = PyPDFLoader("sample_qa.pdf")   # 👈 change with your file
docs = loader.load()

# ----------- Split text into chunks -----------
splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
documents = splitter.split_documents(docs)

# ----------- Create embeddings + FAISS vectorstore -----------
embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
vectorstore = FAISS.from_documents(documents, embeddings)

# ----------- Create retriever -----------
retriever = vectorstore.as_retriever()

# ----------- Setup Gemini LLM -----------
llm = ChatGoogleGenerativeAI(model="gemini-2.0-flash", temperature=0)

# Optional: Custom prompt template for better answers
prompt = ChatPromptTemplate.from_template(
    """
    You are an AI assistant. Use the context below to answer the question.
    If the answer is not in the context, just say you don't know.

    Context:
    {context}

    Question: {input}
    Answer:
    """
)

# ----------- Create document chain + retrieval chain -----------
doc_chain = create_stuff_documents_chain(llm, prompt)
qa_chain = create_retrieval_chain(retriever, doc_chain)

# ----------- Ask Questions -----------
while True:
    query = input("\nAsk a question about your file (or type 'exit'): ")
    if query.lower() == "exit":
        break
    
    result = qa_chain.invoke({"input": query})
    print("\nAnswer:", result["answer"])
