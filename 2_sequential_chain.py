from langchain_huggingface import ChatHuggingFace, HuggingFacePipeline
from langchain_core.prompts import PromptTemplate
from langchain_openai import ChatOpenAI 
from dotenv import load_dotenv
# from langchain_core.output_parsers import StrOutputParser
from langchain_core.output_parsers import JsonOutputParser
from langchain.output_parsers import StructuredOutputParser, ResponseSchema
from langchain_core.output_parsers import PydanticOutputParser
from langchain_core.output_parsers import StrOutputParser 

from pydantic import BaseModel, Field
import os


load_dotenv()

model = ChatOpenAI(
    base_url="https://models.github.ai/inference",  # GitHub Models endpoint
    api_key=os.environ["GITHUB_TOKEN"],             # PAT from .env
    model="gpt-4o-mini",                            # or "gpt-4o", "gpt-3.5-turbo"
    temperature=0.7
)


prompt1 = PromptTemplate(
    template="Generate a detailed report on {topic}.",
    input_variables=["topic"]
)
prompt2 = PromptTemplate(
    template="Generate a 5 point summary of the following text \n {text}" ,
    input_variables=["text"]
)

parser = StrOutputParser()
chain= prompt1 | model |parser | prompt2 | model | parser

result=chain.invoke({"topic":" unemployment in India"})
print(result)