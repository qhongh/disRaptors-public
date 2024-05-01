import streamlit as st
from disraptors.rag.reddit_ingestor import PineConeBaseIngestor, NAMESPACE
from disraptors.rag.pulse import Pulse
from functools import cache

@cache
def get_pulse_api():
    pi = PineConeBaseIngestor(index_name="disraptors-w", namespace=NAMESPACE)
    retriever = pi.vector_store
    p = Pulse(retriever=retriever, refresh=False)
    return p

top_k = 50
alpha = 0.7
pulse = get_pulse_api()

if "messages" not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

if query := st.chat_input("How I can help you?"):
    st.chat_message("user").markdown(query)
    st.session_state.messages.append({"role": "user", "content": query})

    response = pulse.chat(query=query)
    with st.chat_message("assistant"):
        st.markdown(response)
    st.session_state.messages.append({"role": "assistant", "content": response})