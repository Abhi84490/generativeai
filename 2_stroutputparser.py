from langchain_huggingface import ChatHuggingFace, HuggingFacePipeline
from langchain_core.prompts import PromptTemplate
from langchain_openai import ChatOpenAI 
from dotenv import load_dotenv
from langchain_core.output_parsers import StrOutputParser
import os

# Load environment variables
load_dotenv()

# ✅ Use your GitHub token (stored in .env as GITHUB_TOKEN)
# GitHub Models works with OpenAI-compatible APIs, so ChatOpenAI can be pointed there
model = ChatOpenAI(
    base_url="https://models.github.ai/inference",  # GitHub Models endpoint
    api_key=os.environ["GITHUB_TOKEN"],             # PAT from .env
    model="gpt-4o-mini",                            # or "gpt-4o", "gpt-3.5-turbo"
    temperature=0.7
)

# 1st prompt → detailed report
template1 = PromptTemplate(
    template="Write a detailed report on {topic}",
    input_variables=["topic"]
)

# 2nd prompt → short summary
template2 = PromptTemplate(
    template="Write a 5 line summary on the following text:\n\n{text}",
    input_variables=["text"]
)

parser = StrOutputParser()

# Build the chain
chain = template1 | model | parser | template2 | model | parser

# Run the chain
result = chain.invoke({"topic": "Black holes"})
print(result)
