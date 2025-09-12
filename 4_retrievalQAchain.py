from langchain.document_loaders import PyPDFLoader, TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_core.embeddings import OpenAIEmbeddings
from langchain_core.vectorstores import FAISS
from langchain.chains import RetrievalQA
from langchain.llms import ChatOpenAI
from langchain.llms import openai
import os
from dotenv import load_dotenv  
load_dotenv()  # take environment variables from .env.
loader=TextLoader('text.txt')
documents=loader.load()
text_splitter=RecursiveCharacterTextSplitter(chunk_size=1000,chunk_overlap=0)
docs=text_splitter.split_documents(documents)

vectorstore=FAISS.from_documents(docs,OpenAIEmbeddings())
retriever=vectorstore.as_retriever()
llm=ChatOpenAI(
    base_url="https://models.github.ai/inference",  # GitHub Models endpoint
    api_key=os.environ["GITHUB_TOKEN"],             # PAT from .env
    model="gpt-4o-mini",                            # or "gpt-4o", "gpt-3.5-turbo"
    temperature=0.7
)
qa_chain=RetrievalQA.from_chain_type(llm=llm,retriever=retriever)

query="What are the key takeaways from the document?"
answer=qa_chain.run(query)
print("Answer:",answer)