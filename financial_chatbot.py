import pandas as pd
import streamlit as st
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

file_name = "D:/GEN AI/Langchain/Demo Projects/Financial Data Analysis/data/financial_data.csv"

llm = ChatGroq(model_name="llama3-8b-8192")

# Create the CSV agent
agent = create_csv_agent(
    llm,
    file_name,
    verbose=True,
    agent_type=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
    allow_dangerous_code=True
)

st.subheader("💬 Ask a question about your data")
st.write("You can ask questions like:")
query = st.text_input("Type your question here:")
show_steps = st.checkbox("🧠 Show intermediate reasoning steps")

if query:
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
   
