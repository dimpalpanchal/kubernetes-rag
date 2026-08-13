import streamlit as st
import requests
import os
import uuid

BACKEND_URL = os.getenv("BACKEND_URL", "http://localhost:8000")

st.set_page_config(
    page_title="Kubernetes AI Assistant",
    page_icon="☸️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for ChatGPT style interface
st.markdown("""
<style>
    /* Dark theme overrides */
    .stApp {
        background-color: #171717;
        color: #ececec;
        font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif;
    }
    
    [data-testid="stSidebar"] {
        background-color: #171717 !important;
        border-right: 1px solid #2f2f2f !important;
    }
    
    /* Hero Title */
    .hero-container {
        display: flex;
        flex-direction: column;
        align-items: center;
        justify-content: center;
        text-align: center;
        padding-top: 12vh;
        padding-bottom: 4vh;
    }
    
    .hero-title {
        font-size: 2.2rem;
        font-weight: 600;
        color: #f3f3f3;
        margin-bottom: 8px;
        letter-spacing: -0.02em;
    }

    .hero-subtitle {
        font-size: 1rem;
        color: #9b9b9b;
        margin-bottom: 2rem;
    }

    /* Card buttons */
    div.stButton > button {
        border-radius: 12px;
        border: 1px solid #333333;
        background-color: #212121;
        color: #e3e3e3;
        transition: all 0.2s ease-in-out;
        font-size: 0.92rem;
        padding: 10px 14px;
    }
    
    div.stButton > button:hover {
        background-color: #2f2f2f;
        border-color: #4f4f4f;
        color: #ffffff;
        transform: translateY(-1px);
    }
    
    /* Hide top padding in streamlit */
    .block-container {
        padding-top: 2rem !important;
        max-width: 900px !important;
    }
    
    /* Custom expander styling */
    .stExpander {
        border: 1px solid #333333 !important;
        border-radius: 10px !important;
        background-color: #212121 !important;
    }
</style>
""", unsafe_allow_html=True)

if "token" not in st.session_state:
    st.session_state.token = None
if "session_id" not in st.session_state:
    st.session_state.session_id = str(uuid.uuid4())
if "messages" not in st.session_state:
    st.session_state.messages = []
if "preset_prompt" not in st.session_state:
    st.session_state.preset_prompt = None

def login():
    st.markdown("<div class='hero-container'><h1 class='hero-title'>Welcome Back</h1><p class='hero-subtitle'>Login to Kubernetes AI Assistant</p></div>", unsafe_allow_html=True)
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        with st.form("login_form"):
            username = st.text_input("Username")
            password = st.text_input("Password", type="password")
            submit = st.form_submit_button("Login", use_container_width=True)
            
            if submit:
                res = requests.post(
                    f"{BACKEND_URL}/auth/login",
                    data={"username": username, "password": password}
                )
                if res.status_code == 200:
                    st.session_state.token = res.json().get("access_token")
                    st.session_state.session_id = str(uuid.uuid4())
                    st.session_state.messages = []
                    st.success("Logged in successfully!")
                    st.rerun()
                else:
                    st.error("Login failed. Check your credentials.")

def signup():
    st.markdown("<div class='hero-container'><h1 class='hero-title'>Create Account</h1><p class='hero-subtitle'>Sign up for Kubernetes AI Assistant</p></div>", unsafe_allow_html=True)
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        with st.form("signup_form"):
            username = st.text_input("Username")
            password = st.text_input("Password", type="password")
            submit = st.form_submit_button("Sign Up", use_container_width=True)
            
            if submit:
                res = requests.post(
                    f"{BACKEND_URL}/auth/signup",
                    json={"username": username, "password": password}
                )
                if res.status_code == 201:
                    st.success("Signed up successfully! You can now log in.")
                else:
                    st.error("Sign up failed. Username might be taken.")

def load_session_messages(session_id: str):
    headers = {"Authorization": f"Bearer {st.session_state.token}"}
    res = requests.get(f"{BACKEND_URL}/chat/history/{session_id}", headers=headers)
    messages = []
    if res.status_code == 200:
        history = res.json()
        for msg in history:
            messages.append({"role": msg["role"], "content": msg["content"]})
    return messages

def fetch_sessions():
    headers = {"Authorization": f"Bearer {st.session_state.token}"}
    res = requests.get(f"{BACKEND_URL}/chat/sessions", headers=headers)
    if res.status_code == 200:
        return res.json()
    return []

def delete_session(session_id: str):
    headers = {"Authorization": f"Bearer {st.session_state.token}"}
    requests.delete(f"{BACKEND_URL}/chat/sessions/{session_id}", headers=headers)
    if st.session_state.session_id == session_id:
        st.session_state.session_id = str(uuid.uuid4())
        st.session_state.messages = []
    st.rerun()

def send_chat_message(prompt: str):
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
                    with st.expander("📚 Sources"):
                        for i, src in enumerate(sources):
                            st.markdown(f"**Chunk {i+1}** (Source: {(src.get('metadata_') or {}).get('source', 'Unknown')}):")
                            st.text(src["content"])
                st.session_state.messages.append({
                    "role": "assistant",
                    "content": answer,
                    "sources": sources
                })
                st.rerun()
            else:
                st.error(f"Failed to get response: {res.status_code} - {res.text}")

def chat_interface():
    # --- SIDEBAR UI ---
    with st.sidebar:
        st.markdown("### ☸️ Kubernetes AI")
        
        if st.button("➕ New chat", use_container_width=True, type="primary"):
            st.session_state.session_id = str(uuid.uuid4())
            st.session_state.messages = []
            st.session_state.preset_prompt = None
            st.rerun()
            
        st.markdown("---")
        st.caption("Recent Chats")
        
        sessions = fetch_sessions()
        if not sessions:
            st.caption("No chat history yet.")
        else:
            for s in sessions:
                s_id = s["session_id"]
                title = s.get("title", "New Conversation")
                is_active = (s_id == st.session_state.session_id)
                
                col1, col2 = st.columns([0.83, 0.17])
                with col1:
                    btn_label = f"💬 {title}" if not is_active else f"📌 {title}"
                    if st.button(btn_label, key=f"sess_{s_id}", use_container_width=True):
                        st.session_state.session_id = s_id
                        st.session_state.messages = load_session_messages(s_id)
                        st.session_state.preset_prompt = None
                        st.rerun()
                with col2:
                    if st.button("🗑️", key=f"del_{s_id}", help="Delete chat"):
                        delete_session(s_id)

        st.markdown("---")
        if st.button("🚪 Logout", use_container_width=True):
            st.session_state.token = None
            st.session_state.messages = []
            st.session_state.session_id = str(uuid.uuid4())
            st.session_state.preset_prompt = None
            st.rerun()

    # Load history if current messages state is empty
    if not st.session_state.messages:
        history_msgs = load_session_messages(st.session_state.session_id)
        if history_msgs:
            st.session_state.messages = history_msgs

    # --- MAIN CHAT LANDING VS MESSAGES ---
    if not st.session_state.messages:
        # ChatGPT Landing Hero Screen
        st.markdown("""
        <div class='hero-container'>
            <h1 class='hero-title'>What can I help with today?</h1>
            <p class='hero-subtitle'>Ask technical questions, debug errors, or explore Kubernetes docs</p>
        </div>
        """, unsafe_allow_html=True)
        
        # Suggestion Cards Grid (2x2)
        c1, c2 = st.columns(2)
        with c1:
            if st.button("🚢 Deploy a Web App\nHow to create a Deployment and Service?", use_container_width=True):
                st.session_state.preset_prompt = "How to create a Kubernetes Deployment and Service?"
                st.rerun()
            if st.button("⚖️ Configure Autoscaling\nHow to set up Horizontal Pod Autoscaler (HPA)?", use_container_width=True):
                st.session_state.preset_prompt = "How to set up Horizontal Pod Autoscaler (HPA)?"
                st.rerun()
        with c2:
            if st.button("🔍 Troubleshoot Pod Errors\nHow to debug CrashLoopBackOff status in Kubernetes?", use_container_width=True):
                st.session_state.preset_prompt = "How to debug CrashLoopBackOff status in Kubernetes?"
                st.rerun()
            if st.button("🛡️ Cluster Security\nBest practices for Kubernetes RBAC and NetworkPolicies", use_container_width=True):
                st.session_state.preset_prompt = "What are the best practices for Kubernetes RBAC and NetworkPolicies?"
                st.rerun()

    else:
        # Active Chat Conversation Screen
        for message in st.session_state.messages:
            with st.chat_message(message["role"]):
                st.markdown(message["content"])
                if "sources" in message and message["sources"]:
                    with st.expander("📚 Sources"):
                        for i, src in enumerate(message["sources"]):
                            st.markdown(f"**Chunk {i+1}** (Source: {(src.get('metadata_') or {}).get('source', 'Unknown')}):")
                            st.text(src["content"])

    # Handle preset prompt from suggestion card click
    if st.session_state.preset_prompt:
        prompt_to_send = st.session_state.preset_prompt
        st.session_state.preset_prompt = None
        send_chat_message(prompt_to_send)

    # Standard Chat Input Box
    if prompt := st.chat_input("Ask anything about Kubernetes..."):
        send_chat_message(prompt)

if st.session_state.token:
    chat_interface()
else:
    tab1, tab2 = st.tabs(["Login", "Sign Up"])
    with tab1:
        login()
    with tab2:
        signup()
