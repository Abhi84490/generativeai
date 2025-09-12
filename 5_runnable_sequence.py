from langchain_openai import ChatOpenAI
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser   
from dotenv import load_dotenv
from langchain.schema.runnable import RunnableSequence
import os
load_dotenv()

prompt1=PromptTemplate(
    template='Write a joke about {topic}',
    input_variables=['topic']
)


model = ChatOpenAI(
    base_url="https://models.github.ai/inference",  # GitHub Models endpoint
    api_key=os.environ["GITHUB_TOKEN"],             # PAT from .env
    model="gpt-4o-mini",                            # or "gpt-4o", "gpt-3.5-turbo"
    temperature=0.7
)

parser=StrOutputParser()


prompt2=PromptTemplate(
    template="Explain the following joke -{text}",
    input_variables=["text"]

)


chain=RunnableSequence(prompt1,model,parser,prompt2,model,parser)
print(chain.invoke({"topic":"AI"}))