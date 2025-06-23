# app.py
import streamlit as st
from database_utils import close_persistent_connection

st.set_page_config(
    page_title="AI Resume Analyzer",
    page_icon="📄",
    layout="wide"
)

st.sidebar.title("Navigation")
page = st.sidebar.radio("Go to", ["Client Portal", "Admin Portal"])

if page == "Client Portal":
    close_persistent_connection()
    from client_page import show_client_page
    show_client_page()
else:
    from admin_page import show_admin_page
    show_admin_page()