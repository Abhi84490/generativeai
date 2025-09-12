from langchain_openai import ChatOpenAI
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser   
from dotenv import load_dotenv
from langchain.schema.runnable import RunnableSequence, RunnableParallel, RunnablePassthrough, RunnableLambda,RunnableBranch
import os
load_dotenv()


prompt1=PromptTemplate(
    template="Write a detail report on {topic}",
    input_variables=["topic"]
)

prompt2=PromptTemplate(
    template="Summarize the following text \n{text}",
    input_variables=["text"]
)


model = ChatOpenAI(
    base_url="https://models.github.ai/inference",  # GitHub Models endpoint
    api_key=os.environ["GITHUB_TOKEN"],             # PAT from .env
    model="gpt-4o-mini",                            # or "gpt-4o", "gpt-3.5-turbo"
    temperature=0.7
)

parser=StrOutputParser()

# report_gen_chain=RunnableSequence(prompt1,model,parser)
report_gen_chain=prompt1| model | parser     
# also use this in place of RunnableSequence full write
branch_chain=RunnableBranch(
    (lambda x: len(x.split())>500, RunnableSequence(prompt2,model,parser)),  # if more than 500 words, summarize
    RunnablePassthrough()

)

final_chain=RunnableSequence(report_gen_chain,branch_chain)
print(final_chain.invoke({"topic": "India vs China"}))
