import os
import streamlit as st
from dotenv import load_dotenv
from langchain_community.tools import WikipediaQueryRun
from langchain_community.utilities import WikipediaAPIWrapper
from langchain_openai import ChatOpenAI
from langgraph.prebuilt import create_react_agent
from langchain_core.messages import HumanMessage

# Load environment variables
load_dotenv()
API_KEY = os.getenv("OPENAI_API_KEY")

# Wikipedia API for external knowledge
api_wrapper = WikipediaAPIWrapper(top_k_results=1, doc_content_chars_max=300)
wikipedia_tool = WikipediaQueryRun(api_wrapper=api_wrapper)

# History Professor Agent
history_professor_llm = ChatOpenAI(temperature=0, api_key=API_KEY, model="gpt-4o-mini")
history_tools = [wikipedia_tool]  # Can be expanded for more history-related tools
history_professor_agent = create_react_agent(history_professor_llm, history_tools)

# Lawyer Agent
lawyer_llm = ChatOpenAI(temperature=0, api_key=API_KEY, model="gpt-4o-mini")
lawyer_tools = [wikipedia_tool]  # You can add law-related APIs here
lawyer_agent = create_react_agent(lawyer_llm, lawyer_tools)

# Streamlit App
st.set_page_config(page_title="Chat with Experts", layout="wide")
st.title("Chat with Experts")

# Sidebar Navigation
page = st.sidebar.radio("Choose an Expert:", ["History Professor", "Lawyer"])

# Chat History
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

def clear_chat():
    st.session_state.chat_history = []

def stop_session():
    st.stop()

# Chat Interface
st.sidebar.button("Clear Chat History", on_click=clear_chat)
st.sidebar.button("End Session", on_click=stop_session)

st.subheader(f"Chat with the {page}")
user_input = st.text_input("Ask a question:")

if user_input:
    if page == "History Professor":
        response = history_professor_agent.invoke({"messages": [HumanMessage(content=user_input)]})
    else:
        response = lawyer_agent.invoke({"messages": [HumanMessage(content=user_input)]})
    
    reply = response["messages"][-1].content
    st.session_state.chat_history.append((user_input, reply))

# Display Chat History
for query, res in st.session_state.chat_history:
    st.write(f"**You:** {query}")
    st.write(f"**{page}:** {res}")

# requirements.txt
requirements_txt = """\nstreamlit\ndotenv\nlangchain\nlangchain-community\nlanggraph\nopenai\nlangchain_openai\nwikipedia\n"""
with open("requirements.txt", "w") as f:
    f.write(requirements_txt)

st.success("Setup Complete! Run 'streamlit run app.py' to launch the chatbot.")
