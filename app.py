import streamlit as st
from run_rag import main

st.title('Biology Question RAG App')

query = st.text_input("Enter your question:")
if st.button('Answer'):
    args = ['--query', query]
    retrieved_texts = main(args)
    st.write("Answer:", retrieved_texts )