"""Streamlit dashboard application entry point."""


def main() -> None:
    import streamlit as st

    st.set_page_config(page_title="IP Risk Agentic RAG", layout="wide")
    st.title("基于 Agentic RAG 的跨境电商知识产权风险初筛系统")
