from langchain.openai import ChatOpenAI
from langchain_core.prompts import PromptTemplate

from dotenv import load_dotenv
import os
mode = ChatOpenAI(
    base_url="https://models.github.ai/inference",  # GitHub Models endpoint
    api_key=os.environ["GITHUB_TOKEN"],             # PAT from .env
    model="gpt-4o-mini",                            # or "gpt-4o", "gpt-3.5-turbo"
    temperature=0.7
)
prompt=PromptTemplate(
    input_variables=["topic"],
    template="Suggest a catchy blog title about {topic}"
)

topic=input("Enter a topic: ")

formatted_prompt=prompt.format(topic=topic)

blog_title=mode.invoke(formatted_prompt)
print("Generated Blog Title: ", blog_title)