import streamlit as st
import requests
import os
import uuid

BACKEND_URL = os.getenv("BACKEND_URL", "http://localhost:8000")

st.set_page_config(page_title="Kubernetes RAG Assistant", layout="wide")

if "token" not in st.session_state:
    st.session_state.token = None
if "session_id" not in st.session_state:
    st.session_state.session_id = str(uuid.uuid4())
if "messages" not in st.session_state:
    st.session_state.messages = []

def login():
    st.title("Login to Kubernetes RAG Assistant")
    with st.form("login_form"):
        username = st.text_input("Username")
        password = st.text_input("Password", type="password")
        submit = st.form_submit_button("Login")
        
        if submit:
            res = requests.post(
                f"{BACKEND_URL}/auth/login",
                data={"username": username, "password": password}
            )
            if res.status_code == 200:
                st.session_state.token = res.json().get("access_token")
                st.success("Logged in successfully!")
                st.rerun()
            else:
                st.error("Login failed. Check your credentials.")

def signup():
    st.title("Sign Up")
    with st.form("signup_form"):
        username = st.text_input("Username")
        password = st.text_input("Password", type="password")
        submit = st.form_submit_button("Sign Up")
        
        if submit:
            res = requests.post(
                f"{BACKEND_URL}/auth/signup",
                json={"username": username, "password": password}
            )
            if res.status_code == 201:
                st.success("Signed up successfully! You can now log in.")
            else:
                st.error("Sign up failed. Username might be taken.")

def chat_interface():
    st.title("Kubernetes Docs Assistant")
    
    st.sidebar.button("New Session", on_click=lambda: st.session_state.update(session_id=str(uuid.uuid4()), messages=[]))
    
    if st.sidebar.button("Logout"):
        st.session_state.token = None
        st.rerun()
    
    # Load history if empty
    if not st.session_state.messages:
        headers = {"Authorization": f"Bearer {st.session_state.token}"}
        res = requests.get(f"{BACKEND_URL}/chat/history/{st.session_state.session_id}", headers=headers)
        if res.status_code == 200:
            history = res.json()
            for msg in history:
                st.session_state.messages.append({"role": msg["role"], "content": msg["content"]})
    
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
            if "sources" in message and message["sources"]:
                with st.expander("Sources"):
                    for i, src in enumerate(message["sources"]):
                        st.markdown(f"**Chunk {i+1}** (Source: {(src.get('metadata_') or {}).get('source', 'Unknown')}):")
                        st.text(src["content"])

    if prompt := st.chat_input("Ask about Kubernetes..."):
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)
            
        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                headers = {"Authorization": f"Bearer {st.session_state.token}"}
                payload = {"session_id": st.session_state.session_id, "message": prompt}
                res = requests.post(f"{BACKEND_URL}/chat/", json=payload, headers=headers)
                
                if res.status_code == 200:
                    data = res.json()
                    answer = data.get("response", "")
                    sources = data.get("sources", [])
                    st.markdown(answer)
                    if sources:
                        with st.expander("Sources"):
                            for i, src in enumerate(sources):
                                st.markdown(f"**Chunk {i+1}** (Source: {(src.get('metadata_') or {}).get('source', 'Unknown')}):")
                                st.text(src["content"])
                    st.session_state.messages.append({
                        "role": "assistant",
                        "content": answer,
                        "sources": sources
                    })
                else:
                    st.error(f"Failed to get response: {res.status_code} - {res.text}")

if st.session_state.token:
    chat_interface()
else:
    tab1, tab2 = st.tabs(["Login", "Sign Up"])
    with tab1:
        login()
    with tab2:
        signup()
