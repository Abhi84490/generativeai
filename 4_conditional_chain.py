from langchain_huggingface import ChatHuggingFace, HuggingFacePipeline
from langchain_core.prompts import PromptTemplate
from langchain_openai import ChatOpenAI 
from dotenv import load_dotenv
# from langchain_core.output_parsers import StrOutputParser
from langchain_core.output_parsers import JsonOutputParser
from langchain.output_parsers import StructuredOutputParser, ResponseSchema
from langchain_core.output_parsers import PydanticOutputParser
from langchain_core.output_parsers import StrOutputParser 
from langchain.schema.runnable import RunnableParallel, RunnableBranch, RunnableLambda
from pydantic import BaseModel, Field
from typing import Literal

from pydantic import BaseModel, Field
import os


load_dotenv()

mode = ChatOpenAI(
    base_url="https://models.github.ai/inference",  # GitHub Models endpoint
    api_key=os.environ["GITHUB_TOKEN"],             # PAT from .env
    model="gpt-4o-mini",                            # or "gpt-4o", "gpt-3.5-turbo"
    temperature=0.7
)

parser=StrOutputParser()


class Feedback(BaseModel):
    # feedback: str = Field(..., description="The feedback text to be classified")
    sentiment: Literal["positive", "negative"] = Field(..., description="The sentiment of the feedback, either 'positive' or 'negative'") 

parser2=PydanticOutputParser(pydantic_object=Feedback)
prompt1=PromptTemplate(
    template="Classify the sentiment of the feedback text into positive or negative \n {feedback} \n {format_instructions}" ,
    input_variables=["feedback"],
    partial_variables={"format_instructions": parser2.get_format_instructions()} 
)

classifier_chain= prompt1 | mode | parser2

prompt2=PromptTemplate(
    template="Write a appropriate response to this positive feedback: \n {feedback}" ,
    input_variables=["feedback"]
)
prompt3=PromptTemplate(
    template="Write a appropriate response to this negative feedback: \n {feedback}" ,
    input_variables=["feedback"]
)


branch_chain=RunnableBranch(
    (lambda x:x.sentiment=="positive",prompt2 | mode | parser),
    (lambda x:x.sentiment=="negative",prompt3 | mode | parser),
    RunnableLambda(lambda x: "could not find sentiment ",RunnableLambda(lambda x: "could not find sentiment " ))
)


chain=classifier_chain | branch_chain
result=chain.invoke({"feedback": "This is the terrible smart phone "})
print(result)
chain.get_graph().print_ascii()

# result=classifier_chain.invoke({"feedback": "This is the terrible smart phone "}).sentiment
# print(result)