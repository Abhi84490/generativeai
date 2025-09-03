from langchain_huggingface import ChatHuggingFace, HuggingFacePipeline

import streamlit as st


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
# result=model.invoke("what is the capital of india?")
# print(result.content)
st.header("Reseach tool")
user_input=st.text_input("Enter your prompt")

if st.button("Summarize"):
    result=model.invoke(user_input)
    st.write(result.content)