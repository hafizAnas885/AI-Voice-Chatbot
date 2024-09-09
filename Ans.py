from langchain.prompts import (
    ChatPromptTemplate,
    HumanMessagePromptTemplate,
    MessagesPlaceholder,
    SystemMessagePromptTemplate,
)
from langchain_community.chat_message_histories import StreamlitChatMessageHistory
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.schema.output_parser import StrOutputParser
from streamlit_mic_recorder import speech_to_text
from streamlit_TTS import text_to_speech
from gtts.lang import tts_langs
import streamlit as st
import os

st.set_page_config(page_title="AI Voice Assistant", page_icon="🤖")



st.title("AI Voice Assistant 🎙️")
st.subheader("Interact in Urdu with Real-Time Voice Input")

api_key = "..."

prompt = ChatPromptTemplate(
    messages=[
        SystemMessagePromptTemplate.from_template(
            "You are a helpful AI assistant. Please always respond to user queries in Pure Urdu language."
        ),
        MessagesPlaceholder(variable_name="chat_history"),
        HumanMessagePromptTemplate.from_template("{question}"),
    ]
)

msgs = StreamlitChatMessageHistory(key="langchain_messages")

model = ChatGoogleGenerativeAI(model="gemini-1.5-flash", google_api_key=api_key)

chain = prompt | model | StrOutputParser()

chain_with_history = RunnableWithMessageHistory(
    chain,
    lambda session_id: msgs,
    input_messages_key="question",
    history_messages_key="chat_history",
)

langs = tts_langs().keys()

if "text_received" not in st.session_state:
    st.session_state.text_received = ""

st.write("Press the button and start speaking in Urdu:")
if st.button("🎤Ask a Questions through Voice"):
    with st.spinner("Converting Speech To Text..."):
        text = speech_to_text(language="ur", use_container_width=True, just_once=True, key="STT")

    if text:
        st.chat_message("human").write(text)
        with st.chat_message("assistant"):
            message_placeholder = st.empty()
            full_response = ""

            config = {"configurable": {"session_id": "any"}}
            response = chain_with_history.stream({"question": text}, config)

            for res in response:
                full_response += res or ""
                message_placeholder.markdown(full_response + "|")
                message_placeholder.markdown(full_response)

        with st.spinner("Converting Text To Speech..."):
            text_to_speech(text=full_response, language="ur")
    else:
        st.warning("Please press the button and start speaking.")
