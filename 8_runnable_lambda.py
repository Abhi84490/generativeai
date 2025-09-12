from langchain_openai import ChatOpenAI
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser   
from dotenv import load_dotenv
from langchain.schema.runnable import RunnableSequence, RunnableParallel, RunnablePassthrough, RunnableLambda
import os
load_dotenv()


def word_count(text):
    return len(text.split())

# runnable_word_count = RunnableLambda(word_count)
# print(runnable_word_count.invoke("Hello world from LangChain"))    # Output: 4

prompt=PromptTemplate(
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
joke_gen_chain=RunnableSequence(prompt,model,parser)

parallel_chain=RunnableParallel({
    "joke":RunnablePassthrough(),
    "word_count":RunnableLambda(word_count)
})

# parallel_chain=RunnableParallel({
#     "joke":RunnablePassthrough(),
#     "word_count":RunnableLambda(lambda x: len(x.split()))
# })


final_chain=RunnableSequence(joke_gen_chain,parallel_chain)
result=final_chain.invoke({"topic":"cricket"})
final_result="""{}\n word count -{}""".format(result["joke"],result["word_count"])
print(final_result)