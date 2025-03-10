import streamlit as st
import torch
import whisper
import os
from langchain_groq import ChatGroq
from langchain_core.messages import HumanMessage
from dotenv import load_dotenv

# Load environment variables
load_dotenv()
GROQ_API_KEY = os.getenv("GROQ_API_KEY")

# Streamlit UI
st.title("Customer Call Analyzer")
st.write("Upload an audio file (.wav or .mp3) to analyze the conversation.")

uploaded_file = st.file_uploader("Choose an audio file", type=["wav", "mp3"])

if uploaded_file is not None:
    filename = os.path.join("temp_audio", uploaded_file.name)
    os.makedirs("temp_audio", exist_ok=True)
    
    with open(filename, "wb") as f:
        f.write(uploaded_file.getbuffer())
    
    st.success("File uploaded successfully!")
    
    # Load Whisper model
    model = whisper.load_model("small")
    
    # Transcribe audio
    st.write("Transcribing...")
    result = model.transcribe(filename)
    transcription = result["text"]
    st.text_area("Transcription:", transcription, height=200)
    
    # Use LLaMA via Groq API for analysis
    llm = ChatGroq(api_key=GROQ_API_KEY, model_name="llama3-8b-8192")
    
    # Summarization
    summary_prompt = f"""
    Summarize the following customer support conversation:
    {transcription}
    """
    messages = [HumanMessage(content=summary_prompt)]
    summary = llm(messages)
    st.subheader("Summary")
    st.write(summary.content if hasattr(summary, 'content') else summary)
    
    # Generate alternative responses
    response_prompt = f"""
    Extract all responses given by the agent from the following conversation. Identify responses that may not have effectively addressed the customer’s concerns. 
    
    Format the output as follows:
    - Old Response: "<original agent response>"
    - Upgraded Response: "<better alternative>"
    - Reason for improvement: "<explanation>"
    
    Ensure the upgraded response is clear, empathetic, and directly addresses customer concerns. Do not include customer statements in the output.
    
    Conversation:
    {transcription}
    """
    messages = [HumanMessage(content=response_prompt)]
    alternative_response = llm(messages)
    st.subheader("Alternative Response Suggestions")
    st.write(alternative_response.content if hasattr(alternative_response, 'content') else alternative_response)
    
    # Clean up
    os.remove(filename)
