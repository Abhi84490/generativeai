from langchain_openai import OpenAI
from dotenv import load_dotenv

load_dotenv() 

# llm=OpenAI(model='gpt-3.5-turbo-instruct', temperature=0)

llm=OpenAI(
    model='gpt-3.5-turbo-instruct', 
    temperature=0, 
    openai_api_key="sk-5678ijklmnopabcd5678ijklmnopabcd5678ijkl"
)
result=llm.invoke("What is the capital of India?")
print(result)


