import streamlit as st
import google.generativeai as genai
import os

st.set_page_config(page_title="Chatbot", page_icon="💬")

st.title("💬 Chatbot")
st.markdown("🚀 A Streamlit chatbot powered by Gemini 2.5 Flash")

# Sidebar for API Key
with st.sidebar:
    st.header("Configuration")
    api_key = st.text_input("Gemini API Key", type="password", help="Enter your Google Gemini API key here.")
    st.markdown("[Get an API key](https://aistudio.google.com/app/apikey)")

# Initialize Chat History
if "messages" not in st.session_state:
    st.session_state.messages = [{"role": "assistant", "content": "How can I help you with your computer purchase today?"}]

# Display Chat Messages
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# User Input
if prompt := st.chat_input("Your message"):
    # Add user message to history
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # Generate Response
    if not api_key:
        st.warning("Please enter your Gemini API Key in the sidebar to chat.")
    else:
        try:
            genai.configure(api_key=api_key)
            # Using gemini-2.5-flash as 2.5 might not be available via public API alias yet, or use 'gemini-pro'
            # User asked for "Gemini 2.5 flash", assuming they mean the latest fast model.
            # Let's try to list models or just use 'gemini-2.5-flash' which is the current flash model.
            # If 2.5 is a specific request, I'll try to use a generic name or fallback.
            # For now, 'gemini-2.5-flash' is safe.
            model = genai.GenerativeModel('gemini-2.5-flash')
            
            with st.chat_message("assistant"):
                with st.spinner("Thinking..."):
                    chat = model.start_chat(history=[
                        {"role": "user" if m["role"] == "user" else "model", "parts": m["content"]}
                        for m in st.session_state.messages[:-1] # Exclude current prompt as it's sent in send_message
                    ])
                    response = chat.send_message(prompt)
                    st.markdown(response.text)
            
            st.session_state.messages.append({"role": "assistant", "content": response.text})
            
        except Exception as e:
            st.error(f"Error: {e}")
