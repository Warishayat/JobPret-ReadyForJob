import streamlit as st
from Agent.ResumeAnalysis import Resume_Analysis
import json
import matplotlib.pyplot as plt
from streamlit_chat import message as chat_message
from PyPDF2 import PdfReader
import time
import os
from dotenv import load_dotenv
from urllib.parse import parse_qs

# Load environment variables
load_dotenv()

# ---------- Page Config ----------
st.set_page_config(
    page_title="JobPrep Pro", 
    layout="wide",
    page_icon="💼",
    initial_sidebar_state="expanded"
)

# Custom CSS for styling
st.markdown("""
<style>
    :root {
        --primary: #4361ee;
        --secondary: #3f37c9;
        --accent: #4895ef;
        --light: #f8f9fa;
        --dark: #212529;
    }
    
    .stApp {
        background-color: var(--light);
        color: var(--dark);
        min-height: 100vh;
    }
    
    .stButton>button {
        background-color: var(--primary);
        color: white;
        border-radius: 8px;
        padding: 0.5rem 1.5rem;
        border: none;
        font-weight: 500;
        transition: all 0.3s;
    }
    
    .stButton>button:hover {
        background-color: var(--secondary);
        transform: translateY(-2px);
        box-shadow: 0 4px 8px rgba(0,0,0,0.1);
    }
    
    .footer {
        width: 100%;
        background-color: var(--dark);
        color: white;
        text-align: center;
        padding: 1rem;
        margin-top: 2rem;
    }
    
    .card {
        background-color: white;
        border-radius: 12px;
        padding: 1.5rem;
        margin: 1rem 0;
        box-shadow: 0 4px 12px rgba(0,0,0,0.08);
        border-left: 4px solid var(--accent);
    }
    
    .success-card {
        border-left: 4px solid #4cc9f0;
    }
    
    .warning-card {
        border-left: 4px solid #f8961e;
    }
    
    .error-card {
        border-left: 4px solid #f94144;
    }
    
    .chat-container {
        height: 500px;
        overflow-y: auto;
        background-color: white;
        border-radius: 12px;
        padding: 15px;
        margin-bottom: 15px;
        box-shadow: 0 4px 12px rgba(0,0,0,0.08);
    }
    
    .progress-bar {
        height: 10px;
        background-color: #e9ecef;
        border-radius: 5px;
        margin: 1rem 0;
    }
    
    .progress-fill {
        height: 100%;
        border-radius: 5px;
        background-color: var(--primary);
    }
    
    /* Form styling */
    .form-input {
        width: 100%;
        padding: 0.5rem;
        border-radius: 4px;
        border: 1px solid #ddd;
        margin-bottom: 1rem;
    }
    
    .form-textarea {
        width: 100%;
        padding: 0.5rem;
        border-radius: 4px;
        border: 1px solid #ddd;
        height: 150px;
        margin-bottom: 1rem;
    }
    
    .form-submit {
        background-color: var(--primary);
        color: white;
        border: none;
        padding: 0.5rem 1.5rem;
        border-radius: 4px;
        cursor: pointer;
    }
</style>
""", unsafe_allow_html=True)

# ---------- Initialize Session State ----------
def initialize_session():
    session_defaults = {
        'resume_analyzer': Resume_Analysis(),
        'messages': [],
        'analysis_done': False,
        'file_processed': False,
        'generated_questions': [],
        'tailored_docs': {"resume": "", "cover_letter": ""},
        'last_api_call': 0,
        'analysis_result': None,
        'weaknesses': []
    }
    
    for key, value in session_defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value

initialize_session()

# ---------- Page Routing ----------
def show_main_app():
    # ---------- Header ----------
    st.title("💼 JobPrep Pro")
    st.markdown("""
    Your all-in-one career preparation assistant. Analyze resumes, practice interviews, and get personalized feedback.
    """)

    # ---------- Sidebar Configuration ----------
    with st.sidebar:
        st.header("⚙️ Configuration")
        
        # Reset Button
        if st.button("🔄 Reset Session", key="reset_btn", use_container_width=True):
            st.session_state.clear()
            initialize_session()
            st.rerun()
        
        st.markdown("---")
        
        # Analysis Type
        input_type = st.radio(
            "**Select Analysis Type**",
            ["Predefined Skills", "Job Description"],
            index=0
        )
        
        # Role-Skill Mapping
        tech_roles = {
    "Data Scientist": ["Python", "Pandas", "NumPy", "Scikit-learn", "TensorFlow", "SQL", "Data Visualization", "Statistics", "Machine Learning", "Data Wrangling"],
    "Machine Learning Engineer": ["Python", "TensorFlow", "PyTorch", "Scikit-learn", "MLOps", "Docker", "Kubernetes", "AWS/GCP", "CI/CD", "Model Deployment"],
    "Data Engineer": ["Python", "SQL", "ETL", "Apache Spark", "Hadoop", "Airflow", "Data Pipelines", "AWS/GCP/Azure", "Data Warehousing", "Database Design"],
    "Data Analyst": ["SQL", "Python", "Excel", "Tableau", "Power BI", "Data Visualization", "Statistics", "Pandas", "Data Cleaning", "Reporting"],
    "AI Research Scientist": ["Python", "TensorFlow", "PyTorch", "Research Methods", "Neural Networks", "NLP", "Computer Vision", "Mathematics", "Publications", "Algorithm Development"],
    "Frontend Developer": ["JavaScript", "React", "Angular", "Vue", "HTML5", "CSS3", "TypeScript", "Responsive Design", "Web Performance", "UI/UX Principles"],
    "Backend Developer": ["Python", "Java", "Node.js", "Django", "Flask", "Spring", "REST APIs", "Microservices", "Database Design", "Authentication"],
    "Full Stack Developer": ["JavaScript", "React", "Node.js", "Python", "SQL", "REST APIs", "HTML/CSS", "Git", "Docker", "AWS Basics"],
    "DevOps Engineer": ["Docker", "Kubernetes", "CI/CD", "AWS/GCP/Azure", "Terraform", "Ansible", "Linux", "Bash Scripting", "Monitoring", "Infrastructure as Code"],
    "Cloud Engineer": ["AWS", "Azure", "GCP", "Terraform", "Docker", "Kubernetes", "CI/CD", "Networking", "Security", "Cloud Architecture"],
    
    # Specialized Engineering Roles
    "Mobile Developer (Android)": ["Java", "Kotlin", "Android SDK", "MVVM", "Jetpack", "Firebase", "Material Design", "REST APIs", "Performance Optimization"],
    "Mobile Developer (iOS)": ["Swift", "Objective-C", "UIKit", "SwiftUI", "Core Data", "Xcode", "CocoaPods", "REST APIs", "App Store Guidelines"],
    "Game Developer": ["C++", "C#", "Unity", "Unreal Engine", "3D Math", "Physics", "AI for Games", "Multiplayer Networking", "Performance Optimization"],
    "Embedded Systems Engineer": ["C", "C++", "RTOS", "Microcontrollers", "IoT", "Hardware Interfaces", "Firmware", "Debugging", "Communication Protocols"],
    "Computer Vision Engineer": ["Python", "OpenCV", "TensorFlow", "PyTorch", "Image Processing", "Deep Learning", "CNN", "Object Detection", "3D Reconstruction"],
    
    # Security & QA Roles
    "Cybersecurity Engineer": ["Network Security", "Penetration Testing", "Firewalls", "SIEM", "Cryptography", "Python", "Risk Assessment", "Incident Response", "Compliance"],
    "QA Automation Engineer": ["Selenium", "Python/Java", "Test Automation", "JUnit", "TestNG", "CI/CD", "Performance Testing", "API Testing", "Test Planning"],
    "Site Reliability Engineer (SRE)": ["Linux", "Python/Go", "Monitoring", "Incident Management", "Kubernetes", "Docker", "CI/CD", "Capacity Planning", "Chaos Engineering"],
    
    # Web & Design Roles
    "UI/UX Designer": ["Figma", "Adobe XD", "User Research", "Wireframing", "Prototyping", "UI Design", "UX Principles", "Accessibility", "Design Systems"],
    "Web Designer": ["HTML", "CSS", "JavaScript", "Responsive Design", "UI/UX", "Adobe Creative Suite", "WordPress", "SEO Basics", "Performance Optimization"],
    
    # Business & Data Roles
    "Business Intelligence Analyst": ["SQL", "Power BI", "Tableau", "Data Modeling", "ETL", "Dashboarding", "KPI Tracking", "Data Storytelling", "Requirements Gathering"],
    "Data Architect": ["Database Design", "SQL", "NoSQL", "Data Modeling", "ETL", "Data Governance", "Cloud Data Solutions", "Data Security", "Scalability"],
    
    # Emerging Tech Roles
    "Blockchain Developer": ["Solidity", "Ethereum", "Smart Contracts", "Web3.js", "Cryptography", "Distributed Systems", "Node.js", "Security Principles", "Truffle"],
    "AR/VR Developer": ["Unity", "Unreal Engine", "C#", "3D Modeling", "Shader Programming", "Spatial Computing", "User Interaction", "Performance Optimization"],
    "Quantum Computing Engineer": ["Python", "Qiskit", "Linear Algebra", "Quantum Algorithms", "Quantum Mechanics Basics", "Research Skills", "Simulation", "Optimization"],
    
    # IT & Systems Roles
    "Systems Administrator": ["Linux", "Windows Server", "Networking", "Bash/PowerShell", "Virtualization", "Monitoring", "Security", "Troubleshooting", "Automation"],
    "Network Engineer": ["TCP/IP", "Routing", "Switching", "Firewalls", "VPN", "Network Security", "Cisco/Juniper", "Wireless Networks", "Troubleshooting"],
    
    # Specialized Programming Roles
    "Python Developer": ["Python", "Django/Flask", "REST APIs", "OOP", "Testing", "Database", "Performance", "Debugging", "Packaging", "Concurrency"],
    "Java Developer": ["Java", "Spring", "Hibernate", "Microservices", "JUnit", "Design Patterns", "JVM", "Performance Tuning", "Concurrency"],
    "Go Developer": ["Go", "Concurrency", "Microservices", "APIs", "Testing", "Performance", "Cloud", "CLI Tools", "Networking"],
}
        
        # File Upload
        st.markdown("### 📁 Upload Resume")
        uploaded_file = st.file_uploader(
            "Choose a PDF or TXT file",
            type=["pdf", "txt"],
            label_visibility="collapsed"
        )
        
        # Process uploaded file
        if uploaded_file and not st.session_state.file_processed:
            try:
                with st.spinner("Processing your resume..."):
                    if uploaded_file.type == "application/pdf":
                        pdf_reader = PdfReader(uploaded_file)
                        resume_text = "\n".join([page.extract_text() for page in pdf_reader.pages if page.extract_text()])
                    else:
                        resume_text = uploaded_file.getvalue().decode("utf-8")
                    
                    if not resume_text.strip():
                        raise ValueError("The uploaded file appears to be empty")
                    
                    st.session_state.resume_analyzer.resume_text = resume_text
                    st.session_state.resume_analyzer.rag_vectorstore = st.session_state.resume_analyzer.rag_vector_store(resume_text)
                    st.session_state.file_processed = True
                    st.success("Resume processed successfully!")
            except Exception as e:
                st.error(f"Error processing file: {str(e)}")

    # ---------- Main Tabs ----------
    tab1, tab2, tab3, tab4 = st.tabs([
        "📊 Resume Analysis", 
        "💬 Resume Chat", 
        "🎤 Interview Prep", 
        "✨ Tailored Docs"
    ])

    # ---------- Tab 1: Resume Analysis ----------
    with tab1:
        st.header("📊 Resume Analysis")
        
        if not st.session_state.file_processed:
            st.warning("Please upload and process your resume first")
            st.stop()
        
        if input_type == "Predefined Skills":
            selected_role = st.selectbox("Select Target Role", list(tech_roles.keys()), key="role_select")
            st.markdown(f"**Key Skills for {selected_role}:**")
            st.write(", ".join(tech_roles[selected_role]))
        else:
            jd_text = st.text_area("Paste Job Description", height=200, key="jd_text")
        
        if st.button("Analyze Resume", key="analyze_btn"):
            current_time = time.time()
            if current_time - st.session_state.last_api_call < 2:
                st.warning("Please wait a moment before making another request")
                st.stop()
                
            st.session_state.last_api_call = current_time
            
            with st.spinner("Analyzing your resume..."):
                try:
                    if input_type == "Predefined Skills":
                        skills = tech_roles[selected_role]
                    else:
                        if not jd_text.strip():
                            st.warning("Please enter a job description")
                            st.stop()
                        skills = st.session_state.resume_analyzer.extract_skills_from_job_description(jd_text)
                        if not skills:
                            st.error("Could not extract skills from job description")
                            st.stop()
                    
                    analysis_result = st.session_state.resume_analyzer.semantic_skills_analysis(
                        st.session_state.resume_analyzer.resume_text,
                        skills
                    )
                    
                    if not analysis_result:
                        raise ValueError("Analysis returned empty results")
                    
                    analysis_result.setdefault('cutoff_score', 70)
                    st.session_state.analysis_result = analysis_result
                    st.session_state.weaknesses = st.session_state.resume_analyzer.analyze_resume_weakness()
                    st.session_state.analysis_done = True
                    
                except Exception as e:
                    st.error(f"Analysis failed: {str(e)}")
                    st.stop()
            
            if st.session_state.analysis_done and st.session_state.analysis_result:
                analysis_result = st.session_state.analysis_result
                score = analysis_result['overall_score']
                score_color = "#f94144" if score < 50 else "#f8961e" if score < 70 else "#4cc9f0"
                
                # Score Visualization at the TOP
                with st.container():
                    st.markdown("### 📊 Match Score")
                    cols = st.columns([1, 2])
                    
                    with cols[0]:
                        # Pie Chart
                        fig, ax = plt.subplots(figsize=(4, 4))
                        ax.pie(
                            [score, 100 - score],
                            labels=['Match', 'Gap'],
                            colors=[score_color, '#e9ecef'],
                            autopct='%1.1f%%',
                            startangle=90
                        )
                        ax.axis('equal')
                        st.pyplot(fig, use_container_width=True)
                    
                    with cols[1]:
                        # Score Details
                        st.markdown(f"""
                        <div style="margin-left: 1rem;">
                            <h3 style="color: {score_color}; margin-top: 0;">{score}% Match</h3>
                            <div style="width: 100%; background-color: #e9ecef; border-radius: 4px; margin: 0.5rem 0;">
                                <div style="width: {score}%; height: 24px; background-color: {score_color}; border-radius: 4px;"></div>
                            </div>
                            <p>{'🚨 Needs significant improvement' if score < 50 else 
                                '⚠️ Could be stronger' if score < 70 else 
                                '✅ Well-aligned with target role'}</p>
                        </div>
                        """, unsafe_allow_html=True)
                
                # Strengths and Weaknesses below
                with st.container():
                    col1, col2 = st.columns(2)
                    
                    # Strengths Column
                    with col1:
                        with st.expander("💪 Strengths", expanded=True):
                            if analysis_result.get('strengths'):
                                for strength in analysis_result['strengths']:
                                    strength_score = analysis_result['skills_score'].get(strength, 0)
                                    st.markdown(f"""
                                    <div style="border-left: 4px solid #4cc9f0; padding: 0.5rem 1rem; margin-bottom: 1rem; background-color: #f8f9fa;">
                                        <h4 style="margin-top: 0;">{strength} (Score: {strength_score}/10)</h4>
                                        <p>{analysis_result['skills_reasoning'].get(strength, 'No explanation available')}</p>
                                    </div>
                                    """, unsafe_allow_html=True)
                            else:
                                st.info("No significant strengths identified")
                    
                        # Weaknesses Column
                    with col2:
                        with st.expander("🔍 Areas for Improvement", expanded=True):
                            if st.session_state.weaknesses:
                                for weakness in st.session_state.weaknesses:
                                    st.markdown(f"""
                                    <div style="border-left: 4px solid #f94144; 
                                                padding: 1rem; 
                                                margin-bottom: 1rem; 
                                                background-color: #fff5f5;
                                                border-radius: 8px;
                                                box-shadow: 0 2px 8px rgba(0,0,0,0.1);">
                                        <h4 style="margin-top: 0; color: #d90429;">{weakness.get('skill', 'Skill')} 
                                            <span style="font-size: 0.9em; color: #6c757d;">(Score: {weakness.get('score', 0)}/10)</span>
                                        </h4>
                                        <p style="margin-bottom: 0.5rem;"><strong style="color: #6c757d;">Issue:</strong> {weakness.get('detail', 'No details available')}</p>
                                        <p style="margin-bottom: 0.5rem;"><strong style="color: #6c757d;">Suggestions:</strong></p>
                                        <ul style="margin-top: 0.5rem; padding-left: 1.5rem;">
                                            {"".join([f"<li style='margin-bottom: 0.25rem;'>{suggestion}</li>" 
                                                    for suggestion in weakness.get('suggestion', ['No specific suggestions'])[:2]])}
                                        </ul>
                                    </div>
                                    """, unsafe_allow_html=True)
                            
                            # Key Areas Needing Improvement section - no nested expander
                            if score < analysis_result['cutoff_score'] and analysis_result.get('improvement_areas'):
                                st.markdown("""
                                <div style="margin: 1.5rem 0 0.5rem 0; 
                                            padding-bottom: 0.5rem;
                                            border-bottom: 1px solid #e9ecef;">
                                    <h3 style="margin: 0; color: #d90429;">🎯 Key Areas Needing Improvement</h3>
                                </div>
                                """, unsafe_allow_html=True)
                                
                                st.warning("These skills need the most attention based on your target role:")
                                
                                for skill in analysis_result['improvement_areas'][:3]:
                                    skill_score = analysis_result['skills_score'].get(skill, 0)
                                    reasoning = analysis_result['skills_reasoning'].get(skill, "No reasoning provided")
                                    st.markdown(f"""
                                    <div style="margin-bottom: 1.5rem; 
                                                padding: 1.25rem; 
                                                background-color: #fff3cd;
                                                border-radius: 8px;
                                                border-left: 4px solid #ffd60a;
                                                box-shadow: 0 2px 8px rgba(0,0,0,0.1);">
                                        <h4 style="margin-top: 0; color: #6a040f;">
                                            {skill} <span style="font-size: 0.9em; color: #6c757d;">(Score: {skill_score}/10)</span>
                                        </h4>
                                        <p style="margin-bottom: 0;">{reasoning}</p>
                                    </div>
                                    """, unsafe_allow_html=True)
    # ---------- Tab 2: Resume Chat ----------
    with tab2:
        st.header("💬 Chat with Your Resume")
        
        if not st.session_state.file_processed:
            st.warning("Please upload and process your resume first")
            st.stop()
        
        # Chat container
        chat_container = st.container()
        
        # User input with send button
        col1, col2 = st.columns([4, 1])
        with col1:
            user_input = st.text_input(
                "Ask something about your resume:",
                key="chat_input",
                label_visibility="collapsed",
                placeholder="Type your question here..."
            )
        with col2:
            send_button = st.button("🚀 Send", use_container_width=True)
        
        # Process chat input
        if send_button and user_input:
            current_time = time.time()
            if current_time - st.session_state.last_api_call < 2:
                st.warning("Please wait a moment before sending another message")
                st.stop()
                
            st.session_state.last_api_call = current_time
            
            st.session_state.messages.append({"role": "user", "content": user_input})
            with st.spinner("Generating response..."):
                try:
                    response = st.session_state.resume_analyzer.chat_with_resume(user_input)
                    st.session_state.messages.append({"role": "assistant", "content": response})
                except Exception as e:
                    st.error(f"Failed to get response: {str(e)}")
                    st.session_state.messages.append({
                        "role": "assistant", 
                        "content": "Sorry, I couldn't process your request. Please try again later."
                    })
            st.rerun()
        
        # Display chat messages
        with chat_container:
            if not st.session_state.messages:
                st.info("Start chatting with your resume by typing a question above.")
            else:
                for i, msg in enumerate(st.session_state.messages):
                    if msg["role"] == "user":
                        chat_message(
                            msg["content"], 
                            is_user=True, 
                            key=f"user_{i}",
                            avatar_style="identicon"
                        )
                    else:
                        chat_message(
                            msg["content"], 
                            key=f"assistant_{i}",
                            avatar_style="bottts"
                        )

    # ---------- Tab 3: Interview Preparation ----------
    with tab3:
        st.header("🎤 Interview Preparation")
        
        if not st.session_state.file_processed:
            st.warning("Please upload and process your resume first")
            st.stop()
        
        with st.expander("⚙️ Interview Configuration", expanded=True):
            col1, col2, col3 = st.columns(3)
            with col1:
                interview_role = st.selectbox(
                    "Target Role",
                    list(tech_roles.keys()),
                    key="interview_role"
                )
            with col2:
                difficulty = st.selectbox(
                    "Difficulty Level",
                    ["Beginner", "Intermediate", "Advanced"],
                    key="difficulty"
                )
            with col3:
                num_questions = st.slider(
                    "Number of Questions",
                    5, 20, 10,
                    key="num_questions"
                )
        
        if st.button("✨ Generate Questions", key="generate_questions"):
            current_time = time.time()
            if current_time - st.session_state.last_api_call < 5:
                st.warning("Please wait a few seconds before generating more questions")
                st.stop()
                
            st.session_state.last_api_call = current_time
            
            with st.spinner("Generating interview questions..."):
                try:
                    # Generate questions using the class method
                    questions = st.session_state.resume_analyzer.generate_question_answer(
                        interview_type=interview_role,
                        difficulty=difficulty,
                        num_questions=num_questions
                    )
                    
                    # Initialize clean questions list
                    clean_questions = []
                    
                    # Handle string response (raw LLM output)
                    if isinstance(questions, str):
                        # Remove markdown code blocks and whitespace
                        questions = questions.replace("```json", "").replace("```", "").strip()
                        
                        # Try parsing as JSON
                        try:
                            questions = json.loads(questions)
                        except json.JSONDecodeError:
                            # If JSON parsing fails, treat as plain text
                            questions = [{"question": q.strip(), "answer": "", "category": "general"} 
                                       for q in questions.split("\n") if q.strip()]
                    
                    # Ensure we have a list
                    if not isinstance(questions, list):
                        questions = [questions]
                    
                    # Validate and clean each question
                    for q in questions:
                        if isinstance(q, dict):
                            clean_q = {
                                "question": q.get("question", "Question not available").strip(),
                                "answer": q.get("answer", "Answer not available").strip(),
                                "category": q.get("category", "general").lower()
                            }
                        elif isinstance(q, str):
                            clean_q = {
                                "question": q.strip(),
                                "answer": "Refer to experience section",
                                "category": "general"
                            }
                        else:
                            continue
                        
                        # Remove any remaining JSON artifacts
                        for key in ['question', 'answer']:
                            clean_q[key] = clean_q[key].replace('{', '').replace('}', '').replace('"', '')
                        
                        clean_questions.append(clean_q)
                    
                    # Store cleaned questions
                    st.session_state.generated_questions = clean_questions[:num_questions]
                    st.success(f"Generated {len(st.session_state.generated_questions)} interview questions!")
                    
                except Exception as e:
                    st.error(f"Failed to generate questions: {str(e)}")
                    st.session_state.generated_questions = [{
                        "question": "Error generating questions - please try again",
                        "answer": str(e),
                        "category": "error"
                    }]
        
        # Display generated questions
        if st.session_state.generated_questions:
            st.markdown("---")
            st.subheader("Generated Interview Questions")
            
            for i, qa in enumerate(st.session_state.generated_questions):
                with st.expander(f"❓ Question {i+1}: {qa['question'][:50]}...", expanded=(i==0)):
                    st.markdown(f"**Question:** {qa['question']}")
                    st.markdown(f"*Category:* {qa['category'].title()}")
                    st.markdown("---")
                    st.markdown(f"**Suggested Answer:**")
                    st.markdown(qa['answer'] if qa['answer'] else "No answer provided")
                    
                    # Practice area for first question only
                    if i == 0:
                        user_answer = st.text_area(
                            "Type your answer here:",
                            key=f"user_answer_{i}",
                            height=150,
                            placeholder="Enter your response to get feedback..."
                        )
                        if st.button("Get Feedback", key=f"feedback_{i}"):
                            current_time = time.time()
                            if current_time - st.session_state.last_api_call < 5:
                                st.warning("Please wait before requesting more feedback")
                                st.stop()
                                
                            st.session_state.last_api_call = current_time
                            
                            with st.spinner("Analyzing your answer..."):
                                try:
                                    feedback = st.session_state.resume_analyzer.chat_with_resume(
                                        f"""Provide constructive feedback on this interview answer:
                                        Question: {qa['question']}
                                        Answer: {user_answer}
                                        
                                        Focus on:
                                        - Technical accuracy
                                        - Clarity of expression
                                        - Relevance to the role
                                        - Suggested improvements"""
                                    )
                                    st.markdown("**Feedback:**")
                                    st.info(feedback)
                                except Exception as e:
                                    st.error(f"Couldn't get feedback: {str(e)}")

    # ---------- Tab 4: Tailored Documents ----------
    with tab4:
        st.header("✨ Tailored Documents")
        
        if not st.session_state.file_processed:
            st.warning("Please upload and process your resume first")
            st.stop()
        
        # Ensure analysis was completed first
        if not st.session_state.analysis_done:
            st.warning("Please complete Resume Analysis first to identify strengths/weaknesses")
            st.stop()
        
        with st.expander("⚙️ Document Configuration", expanded=True):
            tailored_role = st.selectbox(
                "Target Role",
                list(tech_roles.keys()),
                key="tailor_role"
            )
            enhance_option = st.radio(
                "Enhancement Focus",
                ["Highlight Strengths", "Address Weaknesses", "Both"],
                index=2
            )
        
        if st.button("🛠️ Generate Documents", key="generate_docs"):
            current_time = time.time()
            if current_time - st.session_state.last_api_call < 10:
                st.warning("Please wait before generating more documents")
                st.stop()
                
            st.session_state.last_api_call = current_time
            
            with st.spinner("Creating tailored documents..."):
                try:
                    # Prepare the enhancement focus text
                    enhancement_text = ""
                    if enhance_option == "Highlight Strengths":
                        enhancement_text = "Focus on highlighting these strengths from the analysis: " + \
                                        ", ".join(st.session_state.analysis_result.get('strengths', []))
                    elif enhance_option == "Address Weaknesses":
                        weakness_list = [w['skill'] for w in st.session_state.weaknesses]
                        enhancement_text = "Focus on addressing these weaknesses from the analysis: " + \
                                        ", ".join(weakness_list)
                    else:  # Both
                        strength_text = "Strengths to highlight: " + \
                                      ", ".join(st.session_state.analysis_result.get('strengths', []))
                        weakness_text = "Weaknesses to address: " + \
                                      ", ".join([w['skill'] for w in st.session_state.weaknesses])
                        enhancement_text = f"{strength_text}. {weakness_text}"
                    
                    # Call your existing function with the role and enhancement focus
                    tailored_resume, tailored_cover = st.session_state.resume_analyzer.generate_tailored_documents(
                        role=f"{tailored_role}. {enhancement_text}"
                    )
                    
                    if not tailored_resume.strip() or not tailored_cover.strip():
                        raise ValueError("Document generation returned empty content")
                    
                    st.session_state.tailored_docs = {
                        "resume": tailored_resume,
                        "cover_letter": tailored_cover
                    }
                    st.success("Documents generated successfully!")
                    
                except Exception as e:
                    st.error(f"Failed to generate documents: {str(e)}")
        
        # Display generated documents with strength/weakness focus
        if st.session_state.tailored_docs["resume"]:
            doc_tab1, doc_tab2 = st.tabs(["📄 Tailored Resume", "✉️ Cover Letter"])
            
            with doc_tab1:
                # Show which strengths/weaknesses were incorporated
                st.markdown("**Analysis Features Incorporated:**")
                if enhance_option in ["Highlight Strengths", "Both"] and st.session_state.analysis_result.get('strengths'):
                    st.markdown("✅ **Strengths Highlighted:**")
                    for strength in st.session_state.analysis_result['strengths'][:3]:
                        st.markdown(f"- {strength} (Score: {st.session_state.analysis_result['skills_score'].get(strength, 'N/A')}/10)")
                
                if enhance_option in ["Address Weaknesses", "Both"] and st.session_state.weaknesses:
                    st.markdown("🛠️ **Weaknesses Addressed:**")
                    for weakness in st.session_state.weaknesses[:3]:
                        st.markdown(f"- {weakness['skill']} (Improved with: {', '.join(weakness['suggestion'][:2])}")
                
                st.markdown("---")
                st.markdown(st.session_state.tailored_docs["resume"])
                st.download_button(
                    label="⬇️ Download Resume",
                    data=st.session_state.tailored_docs["resume"],
                    file_name=f"tailored_resume_{tailored_role.lower().replace(' ', '_')}.txt",
                    mime="text/plain"
                )
            
            with doc_tab2:
                st.markdown(st.session_state.tailored_docs["cover_letter"])
                st.download_button(
                    label="⬇️ Download Cover Letter",
                    data=st.session_state.tailored_docs["cover_letter"],
                    file_name=f"cover_letter_{tailored_role.lower().replace(' ', '_')}.txt",
                    mime="text/plain"
                )

# ---------- Contact Page ----------
def show_contact_page():
    st.title("📬 Contact Us")
    st.markdown("""
    <div class="card">
        <h3>Get in Touch</h3>
        <p>Have questions or feedback? We'd love to hear from you!</p>
        
        <div style="margin-top: 2rem;">
            <p><strong>Email:</strong> Warishayat666@gmail.com</p>
            <p><strong>Phone:</strong> +923194758420</p>
            <p><strong>Address:</strong> Islamabad, pakistan</p>
        </div>
        
        <div style="margin-top: 2rem;">
            <h4>Send us a message:</h4>
            <form>
                <div style="margin-bottom: 1rem;">
                    <input type="text" placeholder="Waris Hayat" class="form-input">
                </div>
                <div style="margin-bottom: 1rem;">
                    <input type="email" placeholder="Warishayat666@gmail.com" class="form-input">
                </div>
                <div style="margin-bottom: 1rem;">
                    <textarea placeholder="say something that you want" class="form-textarea"></textarea>
                </div>
                <button type="submit" class="form-submit">Send Message</button>
            </form>
        </div>
    </div>
    """, unsafe_allow_html=True)

# ---------- Privacy Policy Page ----------
def show_privacy_policy():
    st.title("🔒 Privacy Policy")
    st.markdown("""
    <div class="card">
        <h3>Last Updated: January 2023</h3>
        
        <p>JobPrep Pro ("we", "us", or "our") is committed to protecting your privacy. This Privacy Policy explains how we collect, use, and safeguard your information when you use our services.</p>
        
        <h4>1. Information We Collect</h4>
        <p>We may collect personal information including but not limited to:</p>
        <ul>
            <li>Resume/CV content you upload</li>
            <li>Job descriptions you provide</li>
            <li>Contact information</li>
            <li>Usage data and analytics</li>
        </ul>
        
        <h4>2. How We Use Your Information</h4>
        <p>We use the collected information to:</p>
        <ul>
            <li>Provide and improve our services</li>
            <li>Analyze your resume and provide feedback</li>
            <li>Generate tailored documents</li>
            <li>Communicate with you</li>
        </ul>
        
        <h4>3. Data Security</h4>
        <p>We implement appropriate technical and organizational measures to protect your personal data. However, no method of transmission over the Internet is 100% secure.</p>
        
        <h4>4. Changes to This Policy</h4>
        <p>We may update our Privacy Policy from time to time. We will notify you of any changes by posting the new policy on this page.</p>
    </div>
    """, unsafe_allow_html=True)

# ---------- Terms of Service Page ----------
def show_terms_of_service():
    st.title("📜 Terms of Service")
    st.markdown("""
    <div class="card">
        <h3>Last Updated: January 2023</h3>
        
        <p>By accessing or using JobPrep Pro ("Service"), you agree to be bound by these Terms of Service.</p>
        
        <h4>1. Use of Service</h4>
        <p>You may use our Service to:</p>
        <ul>
            <li>Analyze your resume</li>
            <li>Generate interview questions</li>
            <li>Create tailored documents</li>
            <li>Get career advice</li>
        </ul>
        
        <h4>2. User Responsibilities</h4>
        <p>You agree to:</p>
        <ul>
            <li>Provide accurate information</li>
            <li>Not use the Service for illegal purposes</li>
            <li>Not attempt to reverse engineer our systems</li>
        </ul>
        
        <h4>3. Intellectual Property</h4>
        <p>The Service and its original content, features, and functionality are owned by JobPrep Pro and are protected by international copyright, trademark, and other intellectual property laws.</p>
        
        <h4>4. Limitation of Liability</h4>
        <p>JobPrep Pro shall not be liable for any indirect, incidental, special, consequential or punitive damages resulting from your use of the Service.</p>
        
        <h4>5. Governing Law</h4>
        <p>These Terms shall be governed by the laws of the State of California without regard to its conflict of law provisions.</p>
    </div>
    """, unsafe_allow_html=True)

# ---------- Page Routing ----------
query_params = st.query_params
current_page = query_params.get("page", ["home"])[0]

if current_page == "privacy":
    show_privacy_policy()
elif current_page == "terms":
    show_terms_of_service()
elif current_page == "contact":
    show_contact_page()
else:
    show_main_app()

# ---------- Footer ----------
footer = """
<div class="footer">
    <p>© 2023 JobPrep Pro | Made with ❤️ by Waris Hayat (AI-Engineer).</p>
    <div style="margin-top: 0.5rem;">
        <a href="?page=privacy" style="color: #4895ef; margin: 0 10px;">Privacy Policy</a>
        <a href="?page=terms" style="color: #4895ef; margin: 0 10px;">Terms of Service</a>
        <a href="?page=contact" style="color: #4895ef; margin: 0 10px;">Contact Us</a>
    </div>
</div>
"""
st.markdown(footer, unsafe_allow_html=True)