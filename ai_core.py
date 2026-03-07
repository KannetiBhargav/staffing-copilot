from PyPDF2 import PdfReader
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_google_genai import ChatGoogleGenerativeAI

import spacy
import re

# Load the English NLP model
nlp = spacy.load("en_core_web_sm")


def extract_text_from_pdfs(uploaded_files):
    """Reads multiple uploaded PDFs and extracts their raw text."""
    docs = []
    metadatas = []
    
    for file in uploaded_files:
        pdf_reader = PdfReader(file)
        text = ""
        
        for page in pdf_reader.pages:
            extracted = page.extract_text()
            if extracted:
                text += extracted + "\n"
        
        docs.append(text)
        # We save the filename as 'metadata' so the AI knows which text belongs to who
        metadatas.append({"filename": file.name})
        
    return docs, metadatas

def match_resumes(jd_text, uploaded_files, api_key):
    """Converts resumes to vectors and ranks them against the JD."""
    
    
    docs, metadatas = extract_text_from_pdfs(uploaded_files)
    
    
    embeddings = GoogleGenerativeAIEmbeddings(
        model="models/gemini-embedding-001", 
        google_api_key=api_key
    )
    
   
    vector_store = FAISS.from_texts(texts=docs, embedding=embeddings, metadatas=metadatas)
    
    
    results = vector_store.similarity_search_with_score(jd_text, k=len(docs))
    
    return results


def generate_screener(jd_text, uploaded_file, api_key):
    """Generates technical interview questions using Gemini."""
    
    # 1. Extract text from the first uploaded file
    pdf_reader = PdfReader(uploaded_file)
    resume_text = ""
    for page in pdf_reader.pages:
        extracted = page.extract_text()
        if extracted:
            resume_text += extracted + "\n"
            
    # 2. Initialize the Gemini Text Model (We use flash for speed)
    llm = ChatGoogleGenerativeAI(
        model="gemini-2.5-flash", 
        google_api_key=api_key,
        temperature=0.7 # 0.7 gives a good balance of creativity and technical accuracy
    )
    
    # 3. Create the prompt instruction
    prompt = f"""
    You are an expert IT Recruiter and Engineering Manager. 
    Read the following Job Description and the Candidate's Resume.
    Generate 3 highly specific technical interview questions to test if the candidate actually possesses the skills claimed on their resume that match the job description.
    For each question, provide a brief 'Expected Answer' so the recruiter knows what to listen for.

    Job Description:
    {jd_text}

    Candidate Resume:
    {resume_text}
    """
    
    # 4. Send the prompt to Google and get the response
    response = llm.invoke(prompt)
    
    return response.content


def redact_resume(uploaded_file):
    """Redacts PII (Personally Identifiable Information) from a resume."""
    
    # 1. Extract text from the PDF
    pdf_reader = PdfReader(uploaded_file)
    text = ""
    for page in pdf_reader.pages:
        extracted = page.extract_text()
        if extracted:
            text += extracted + "\n"
            
    # 2. Use Regex (Pattern Matching) for foolproof Email and Phone redaction
    # Finds anything that looks like user@domain.com
    text = re.sub(r'[\w\.-]+@[\w\.-]+', '[EMAIL REDACTED]', text)
    # Finds common phone number patterns (e.g., 123-456-7890 or (123) 456 7890)
    text = re.sub(r'\(?\d{3}\)?[-.\s]?\d{3}[-.\s]?\d{4}', '[PHONE REDACTED]', text)

    # 3. Use spaCy NLP to find Human Names
    # We pass the text into the NLP brain
    doc = nlp(text)
    redacted_text = text
    
    # Loop through the entities (objects) spaCy found
    for ent in doc.ents:
        if ent.label_ == "PERSON":
            # If it thinks it's a person's name, replace it
            redacted_text = redacted_text.replace(ent.text, '[NAME REDACTED]')
            
    return redacted_text