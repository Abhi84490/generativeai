from langchai_core.messages import SystemMessage, HumanMessage, AIMessage
# from langchain_huggingface import HuggingFace
from langchain_huggingface import ChatHuggingFace, HuggingFacePipeline



import os
os.environ['HF_HOME'] = 'D:/huggingface_cache'



llm = HuggingFacePipeline.from_model_id(
    model_id="TinyLlama/TinyLlama-1.1B-Chat-v1.0",
    task='text-generation',
    pipeline_kwargs=dict(
        temperature=1.5,
        max_new_tokens=1000,
    )
)
model=ChatHuggingFace(llm=llm)
messages=[
    SystemMessage(content="You are a helpful assistant."),
    HumanMessage(content="Hello, who won the world series in 2020?"),
    # AIMessage(content="The Los Angeles Dodgers won the World Series in 2020.")
]

result=model.invoke(messages)
messages.append(AIMessage(content=result.content))
print(messages)
