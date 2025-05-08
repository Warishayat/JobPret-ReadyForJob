import streamlit as st

# ---------- Page Config ----------
st.set_page_config(page_title="JobPret-BeReadyForJob", layout="wide")
st.title("📄 JobPret-BeReadyForJob")

st.markdown("""
Welcome! Upload your resume and use our intelligent tools to prepare for interviews, analyze resumes, and generate tailored documents.
""")

# ---------- Sidebar: Only Skill Input Type ----------
st.sidebar.header("🧭 Configuration")
input_type = st.sidebar.radio("📌 Resume Analysis Based On", ["Predefined Skills", "Job Description"])

# ---------- Developer-defined role-skill map ----------
tech_roles = {
    "Data Scientist": ["Python", "Pandas", "Scikit-learn", "SQL", "Data Visualization", "Statistics", "Machine Learning"],
    "ML Engineer": ["Python", "TensorFlow", "PyTorch", "MLOps", "Docker", "Kubernetes", "AWS", "CI/CD"],
    "Software Engineer": ["Java", "Python", "Git", "OOP", "SQL", "REST APIs", "Docker"],
    "Frontend Developer": ["JavaScript", "React", "CSS", "HTML", "Web Performance", "Responsive Design"],
    "Backend Developer": ["Node.js", "Express", "MongoDB", "PostgreSQL", "Docker", "API Design"],
    "DevOps Engineer": ["Linux", "AWS", "Terraform", "CI/CD", "Docker", "Kubernetes", "Monitoring"],
}

# ---------- File Upload ----------
uploaded_file = st.sidebar.file_uploader("📎 Upload Resume (PDF or TXT)", type=["pdf", "txt"])


# ---------- Tabs ----------
tab1, tab2, tab3, tab4 = st.tabs([
    "📊 Resume Analysis", 
    "💬 Chat with Resume", 
    "🧠 Interview Preparation", 
    "📝 Tailored Resume & Cover Letter"
])


# ---------- Tab 1: Resume Analysis ----------
with tab1:
    st.subheader("📊 Resume Analysis")

    if input_type == "Predefined Skills":
        selected_role = st.selectbox("🎯 Select Role", list(tech_roles.keys()))
        required_skills = ", ".join(tech_roles[selected_role])
        st.success(f"✅ Required Skills for **{selected_role}**:\n{required_skills}")
    else:
        jd_text = st.text_area("📄 Paste Job Description", height=200)

    if st.button("🔍 Analyze Resume"):
        if uploaded_file:
            st.success("✅ Analysis Complete!")
            st.markdown("**Overall Score:** 87%")
            st.markdown("**Candidate Selected:** ✅ Yes")
            st.markdown("**Strengths:** Python, SQL, Pandas")
            st.markdown("**Missing Skills:** TensorFlow, MLOps")
            st.markdown("---")
            st.markdown("**Sample Weakness:**")
            st.markdown("- **Skill:** TensorFlow — Score: 4/10")
            st.markdown("- **Issue:** No deep learning project mentioned.")
            st.markdown("- **Suggestions:** Add a TensorFlow project using CNNs and training pipeline.")
        else:
            st.warning("📂 Please upload your resume.")


# ---------- Tab 2: Chat with Resume ----------
with tab2:
    st.subheader("💬 Chat with Resume")
    st.info("🔧 Backend coming soon. This tab will allow interactive Q&A with your resume!")


# ---------- Tab 3: Interview Preparation ----------
with tab3:
    st.subheader("🧠 Interview Preparation")
    st.markdown("Select your preferences to generate mock interview questions.")

    col1, col2, col3 = st.columns(3)
    with col1:
        interview_role = st.selectbox("🎯 Role", list(tech_roles.keys()))
    with col2:
        difficulty = st.selectbox("📈 Difficulty", ["Easy", "Medium", "Hard"])
    with col3:
        num_questions = st.slider("❓ Number of Questions", 5, 20, 10)

    if st.button("🎤 Generate Interview Questions"):
        st.success(f"✅ Generated {num_questions} questions for {interview_role} ({difficulty})")
        for i in range(num_questions):
            skill = tech_roles[interview_role][i % len(tech_roles[interview_role])]
            st.markdown(f"**Q{i+1}:** What is {skill}?")
            st.markdown(f"**A:** Explanation about {skill}.")


# ---------- Tab 4: Tailored Resume & Cover Letter ----------
with tab4:
    st.subheader("📝 Tailored Resume & Cover Letter")
    tailored_role = st.selectbox("🎯 Role for Tailoring", list(tech_roles.keys()), key="tailor_role")

    if st.button("📄 Generate Tailored Resume & Cover Letter"):
        st.success(f"✅ Tailored resume and cover letter for {tailored_role} generated!")
        
        st.markdown("### 📄 Tailored Resume")
        st.markdown("**Summary:** Experienced in " + ", ".join(tech_roles[tailored_role]) + " with proven projects and impact.")
        st.markdown("**Skills:**")
        st.markdown(", ".join(tech_roles[tailored_role]))
        st.download_button("⬇️ Download Resume", "Sample tailored resume content", file_name="tailored_resume.txt")

        st.markdown("### ✉️ Cover Letter")
        st.markdown(f"""
Dear Hiring Manager,

I am writing to express my interest in the {tailored_role} position. My background in {', '.join(tech_roles[tailored_role][:3])} and hands-on experience with modern technologies makes me a strong candidate.

Thank you for considering my application.

Sincerely,  
[Your Name]
        """)
        st.download_button("⬇️ Download Cover Letter", "Sample tailored cover letter", file_name="cover_letter.txt")
