import streamlit as st
import google.generativeai as genai
import os
import app_utils

st.set_page_config(page_title="Chatbot", page_icon="💬")

st.title("Agentic Chatbot")
st.markdown("A smart assistant that can predict computer prices using tools.")

with st.expander("Suggested Complex Questions", expanded=False):
    st.markdown("""
    *   **Smart Recommendation**: "Recommend 3 laptops with 16GB RAM and 512GB SSD for under €1200."
    *   **Market Analysis**: "What is the average price of a Dell laptop compared to an HP laptop?"
    *   **Inventory & Price**: "How many Apple laptops do we have, and what is the average price of an Apple laptop?"
    *   **Value Check**: "I have €800. Recommend a laptop with 512GB SSD and tell me if it's cheaper than the average market price."
    """)

with st.expander("Available Tools & Capabilities", expanded=False):
    st.markdown("""
    I have access to the following tools to help answer your questions:
    
    1.  **Price Predictor**: I can estimate the market price for *any* computer configuration (RAM, SSD, GPU, etc.).
    2.  **Product Recommender**: I can find *actual* laptops in our database that match your needs and budget.
    3.  **Market Analyst**: I can calculate average prices for specific specs (e.g., "Average price of 1TB SSD laptops").
    4.  **Inventory Checker**: I can tell you how many laptops we have from a specific brand (e.g., "How many Dells?").
    """)

# Sidebar for API Key
with st.sidebar:
    st.header("Configuration")
    api_key = st.text_input("Gemini API Key", type="password", help="Enter your Google Gemini API key here.")
    st.markdown("[Get an API key](https://aistudio.google.com/app/apikey)")

# Initialize Chat History
if "messages" not in st.session_state:
    st.session_state.messages = [{"role": "assistant", "content": "Hello! I can help you estimate computer prices. Try asking: 'How much is a laptop with 16GB RAM and 512GB SSD?'"}]

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
            
            # Define the tool
            tools = [app_utils.get_price_prediction_for_agent]
            
            # Initialize Model with Tools
            model = genai.GenerativeModel('gemini-2.5-flash', tools=tools)
            
            # Start Chat Session (Automatic function calling handling is supported in chat sessions)
            chat = model.start_chat(enable_automatic_function_calling=True)
            
            # Replay history to set context (excluding the last user prompt which we send next)
            # Note: For simple history replay with tools, we might need to be careful.
            # Ideally, we just send the prompt with history, but `start_chat` manages history.
            # Let's try to just send the prompt for now to keep it simple and robust, 
            # or reconstruct history if needed. 
            # For a robust agent, we often send the full history.
            
            # Simplified approach: Just send the current prompt. 
            # We will construct the history for the chat session
            history_for_gemini = []
            for m in st.session_state.messages[:-1]:
                role = "user" if m["role"] == "user" else "model"
                history_for_gemini.append({"role": role, "parts": [m["content"]]})
                
            # Define the tools
            tools = [
                app_utils.get_price_prediction_for_agent,
                app_utils.recommend_laptops_for_agent,
                app_utils.get_average_price_for_spec,
                app_utils.get_brand_count
            ]
            
            # Initialize Model with Tools
            # We add a system instruction to encourage tool usage with defaults
            system_instruction = "You are a helpful laptop expert agent. If the user asks for a recommendation or price but misses some details (like RAM or SSD), assume reasonable defaults (e.g., 8GB RAM, 256GB SSD) and proceed with the tool call. Do not ask for clarification unless absolutely necessary. Always explain your assumptions."
            model = genai.GenerativeModel('gemini-2.5-flash', tools=tools, system_instruction=system_instruction)
            
            # Start Chat Session (Manual function calling for UI control)
            # We disable automatic function calling to intercept the tool call
            chat = model.start_chat(history=history_for_gemini, enable_automatic_function_calling=False)
            
            with st.chat_message("assistant"):
                # Create a status container to show "thinking" process
                with st.status("Processing...", expanded=True) as status:
                    response = chat.send_message(prompt)
                    
                    # Helper function to find function calls in any part of the response
                    def get_function_call_from_response(resp):
                        if not resp.candidates:
                            return None
                        for part in resp.candidates[0].content.parts:
                            if part.function_call:
                                return part.function_call
                        return None

                    # Loop to handle multiple function calls
                    while True:
                        fc = get_function_call_from_response(response)
                        if not fc:
                            break
                            
                        fn_name = fc.name
                        fn_args = dict(fc.args)
                        
                        status.write(f"**Reasoning:** I need to use a tool to answer your question.")
                        status.write(f"**Tool Call:** `{fn_name}`")
                        status.write(f"**Parameters:** `{fn_args}`")
                        
                        # Execute the tool
                        api_response = "Error: Unknown tool"
                        if fn_name == 'get_price_prediction_for_agent':
                            status.update(label=f"Predicting price...", state="running")
                            api_response = app_utils.get_price_prediction_for_agent(**fn_args)
                        elif fn_name == 'recommend_laptops_for_agent':
                            status.update(label=f"Finding recommendations...", state="running")
                            api_response = app_utils.recommend_laptops_for_agent(**fn_args)
                        elif fn_name == 'get_average_price_for_spec':
                            status.update(label=f"Calculating average price...", state="running")
                            api_response = app_utils.get_average_price_for_spec(**fn_args)
                        elif fn_name == 'get_brand_count':
                            status.update(label=f"Counting laptops...", state="running")
                            api_response = app_utils.get_brand_count(**fn_args)
                        else:
                            status.write("Error: Unknown tool called.")
                            break
                            
                        status.write(f"**Backend Output:** {api_response}")
                        
                        # Send tool output back to model
                        response = chat.send_message(
                            genai.protos.Content(
                                parts=[genai.protos.Part(
                                    function_response=genai.protos.FunctionResponse(
                                        name=fn_name,
                                        response={'result': api_response}
                                    )
                                )]
                            )
                        )
                            
                    status.update(label="Analysis Complete", state="complete", expanded=False)
                
                # Safely display final natural language response
                final_text = ""
                try:
                    final_text = response.text
                except Exception:
                    # If response.text fails, manually extract text parts
                    for part in response.candidates[0].content.parts:
                        if part.text:
                            final_text += part.text + "\n"
                
                if final_text:
                    st.markdown(final_text)
                else:
                    st.warning("No text response generated.")
            
            if final_text:
                st.session_state.messages.append({"role": "assistant", "content": final_text})
            
        except Exception as e:
            st.error(f"Error: {e}")
