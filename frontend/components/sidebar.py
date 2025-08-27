import streamlit as st

def sidebar():
    st.sidebar.header("🔍 Filter Options")
    selected_agent = st.sidebar.selectbox("Select Agent", ["All Agents", "Agent 1", "Agent 2", "Agent 3"])
    return selected_agent

