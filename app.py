import streamlit as st
import pandas as pd
#from langchain_community.llms import Groq
from langchain_groq import ChatGroq
from langchain_experimental.agents.agent_toolkits import create_pandas_dataframe_agent,create_csv_agent
from langchain.agents.agent_types import AgentType
import matplotlib.pyplot as plt
import io
import time

import os
from dotenv import load_dotenv
load_dotenv()

os.environ['GROQ_API_KEY']=os.getenv("GROQ_API_KEY")

st.set_page_config(page_title="Financial Chatbot 💬", layout="wide")
st.title("💰 AI-Powered Financial Data Chatbot")

# Caching the data preprocessing to improve performance
@st.cache
def preprocess_data(df):
    # Example: Optimize data types to save memory and speed up calculations
    df['product'] = df['product'].astype('category')
    df['store'] = df['store'].astype('category')
    return df

# Upload CSV file
uploaded_file = st.file_uploader("📤 Upload your financial CSV file", type=["csv"])

    
if uploaded_file:
    start_time = time.time()
    try:
        # Read first few rows to validate file
        uploaded_file.seek(0)  # Reset file pointer
        df = pd.read_csv(uploaded_file)
        st.subheader("📊 Preview of Your Data")
        st.dataframe(df.head())
    except pd.errors.EmptyDataError:
        st.error("The uploaded file is empty or invalid. Please upload a valid CSV.")
    else:
        # Get Groq API key
        uploaded_file.seek(0)
        llm = ChatGroq(
            model_name="llama3-8b-8192"
        )

        # Create LangChain agent
        # Create LangChain CSV agent
        agent = create_csv_agent(
            llm,
            uploaded_file,  # Pass the file directly
            agent_type=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
            verbose=True,
            allow_dangerous_code=True,
            max_iterations=3,
            max_excecution_time=60,
            return_intermediate_steps=True,
            handle_parsing_errors=True

        )

        # Chat interface
        print("entered chat interface")
        st.subheader("💬 Ask a question about your data")
        query = st.text_input("Type your question here:")
        show_steps = st.checkbox("🧠 Show intermediate reasoning steps")
        if query:
            prompt = (
        "You are a financial analyst AI. Use the uploaded dataset to answer this question clearly:\n"
        f"{query}"
    )
            with st.spinner("Thinking..."):
                try:
                    response = agent.invoke(query)
                    st.success(response['output'])
                    # Optional: Show reasoning trace
                    if show_steps:
                        st.subheader("🪜 Agent Reasoning Steps")
                        for step in response["intermediate_steps"]:
                            st.write(f"- **Action**: {step[0]}")
                            st.write(f"- **Observation**: {step[1]}")
                            st.markdown("---")
                except Exception as e:
                    st.error(f"Error: {e}")

        # Optional: Auto insight generation
        if st.button("🔍 Auto Analyze Dataset"):
            auto_prompt = (
                "Give a high-level summary of this financial dataset. "
                "Mention trends, anomalies, top performing and underperforming entities."
            )
            with st.spinner("Analyzing..."):
                try:
                    result = agent.invoke(auto_prompt)
                    st.info(result['output'])
                    # Optional: Show reasoning trace    
                except Exception as e:
                    st.error(f"Error: {e}")
        end_time = time.time()
        st.write(f"Response Time: {end_time - start_time:.2f} seconds")

