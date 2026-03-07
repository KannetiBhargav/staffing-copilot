
import streamlit as st
import os
from dotenv import load_dotenv
import ai_core

load_dotenv()
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")


st.set_page_config(
    page_title="Intelligent Staffing Copilot",
    page_icon="⚡",
    layout="wide"
)

with st.sidebar:
    st.title("⚙️ Copilot Settings")
    st.markdown("This AI tool automates resume matching, screening, and formatting for IT Recruiters.")
    
    
    if GOOGLE_API_KEY:
        st.success("API Key Loaded Successfully!")
    else:
        st.error("Missing Google API Key. Check your .env file.")
        
    st.markdown("---")
    st.markdown("**Built for Tech Startup Showcase**")


st.title("⚡ Intelligent Staffing Copilot")
st.subheader("Automate your recruitment workflow with AI.")


col1, col2 = st.columns([1, 1])

with col1:
    st.markdown("### 1. Job Description")
    
    jd_text = st.text_area("Paste the full Job Description here:", height=250)

with col2:
    st.markdown("### 2. Candidate Resumes")
    
    uploaded_files = st.file_uploader("Upload PDF Resumes", type=["pdf"], accept_multiple_files=True)

st.markdown("---")


st.markdown("### 3. AI Actions")
action_col1, action_col2, action_col3 = st.columns(3)

with action_col1:
    # Button for Feature 1
    if st.button("🔍 Match Candidates to JD", use_container_width=True):
        if not jd_text or not uploaded_files:
            st.warning("Please provide both a Job Description and at least one Resume.")
        else:
            # st.spinner shows a loading animation while the AI thinks
            with st.spinner("AI is analyzing and matching resumes..."):
                try:
                    # Pass the data to our backend engine
                    results = ai_core.match_resumes(jd_text, uploaded_files, GOOGLE_API_KEY)
                    
                    st.success("Matching Complete!")
                    st.markdown("### 🏆 Top Matches")
                    
                    # FAISS returns a tuple: (Document, L2 Distance Score)
                    # Lower distance score means the resume is mathematically closer to the JD
                    for rank, (doc, score) in enumerate(results, start=1):
                        filename = doc.metadata["filename"]
                        st.write(f"**{rank}. {filename}** *(Distance Score: {score:.4f})*")
                        
                except Exception as e:
                    st.error(f"An error occurred: {e}")

with action_col2:
    # Button for Feature 2
    if st.button("🧠 Generate Tech Screener", use_container_width=True):
        if not jd_text or not uploaded_files:
            st.warning("Please provide both a Job Description and at least one Resume.")
        else:
            with st.spinner("AI is analyzing the profile and writing questions..."):
                try:
                    # For simplicity, we generate questions for the first uploaded resume
                    candidate_file = uploaded_files[0] 
                    
                    # Call our new function
                    screener_text = ai_core.generate_screener(jd_text, candidate_file, GOOGLE_API_KEY)
                    
                    st.success(f"Screener Ready for: {candidate_file.name}")
                    st.markdown("### 🎯 Interview Guide")
                    st.write(screener_text)
                    
                except Exception as e:
                    st.error(f"An error occurred: {e}")
with action_col3:
    # Button for Feature 3
    if st.button("🛡️ Blind Format (Redact PII)", use_container_width=True):
        if not uploaded_files:
            st.warning("Please upload a resume to redact.")
        else:
            with st.spinner("NLP is scanning and redacting sensitive information..."):
                try:
                    # We process the first uploaded resume
                    candidate_file = uploaded_files[0] 
                    
                    # Pass it to our NLP engine
                    redacted_text = ai_core.redact_resume(candidate_file)
                    
                    st.success(f"Redaction Complete for: {candidate_file.name}")
                    st.markdown("### 🛡️ Blind Resume (Safe to Send)")
                    
                    # Display the cleaned text in a scrollable text box
                    st.text_area("Anonymized Text:", redacted_text, height=400)
                    
                except Exception as e:
                    st.error(f"An error occurred: {e}")