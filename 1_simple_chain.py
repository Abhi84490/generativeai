from langchain_huggingface import ChatHuggingFace, HuggingFacePipeline
from langchain_core.prompts import PromptTemplate
from langchain_openai import ChatOpenAI 
from dotenv import load_dotenv
# from langchain_core.output_parsers import StrOutputParser
from langchain_core.output_parsers import JsonOutputParser
from langchain.output_parsers import StructuredOutputParser, ResponseSchema
from langchain_core.output_parsers import PydanticOutputParser
from pydantic import BaseModel, Field
import os


load_dotenv()

model = ChatOpenAI(
    base_url="https://models.github.ai/inference",  # GitHub Models endpoint
    api_key=os.environ["GITHUB_TOKEN"],             # PAT from .env
    model="gpt-4o-mini",                            # or "gpt-4o", "gpt-3.5-turbo"
    temperature=0.7
)


# Define response schemas
schemas = [
    ResponseSchema(name="fact1", description="First interesting fact"),
    ResponseSchema(name="fact2", description="Second interesting fact"),
    ResponseSchema(name="fact3", description="Third interesting fact"),
    ResponseSchema(name="fact4", description="Fourth interesting fact"),
    ResponseSchema(name="fact5", description="Fifth interesting fact"),
]

# ✅ Correct initialization
parser = StructuredOutputParser.from_response_schemas(schemas)

# Get format instructions
format_instructions = parser.get_format_instructions()

# Prompt
prompt = PromptTemplate(
    template="Generate 5 interesting facts about {topic}.\n{format_instructions}",
    input_variables=["topic"],
    partial_variables={"format_instructions": format_instructions}
)

# Chain
chain = prompt | model | parser

# Run
result = chain.invoke({"topic": "cricket"})
print(result)

chain.get_graph().print_ascii()