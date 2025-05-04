import streamlit as st

st.set_page_config(page_title="AI Resume Analyzer", layout="wide")
st.title("📄 AI Resume Analyzer (UI Demo)")

st.markdown("""
Upload your resume and either provide a job description or choose from predefined tech skills.
This demo only displays the interface without running backend logic.
""")

# --- File Upload ---
uploaded_file = st.file_uploader("📎 Upload Resume (PDF or TXT)", type=["pdf", "txt"])

# --- Job Description or Predefined Skills ---
st.sidebar.header("🎯 Skills Configuration")

input_type = st.sidebar.radio("How do you want to analyze your resume?", ["Predefined Skills", "Job Description"])

# Predefined skills per job category
tech_roles = {
    "Data Scientist": ["Python", "Pandas", "Scikit-learn", "SQL", "Data Visualization", "Statistics", "Machine Learning"],
    "ML Engineer": ["Python", "TensorFlow", "PyTorch", "MLOps", "Docker", "Kubernetes", "AWS", "CI/CD"],
    "Software Engineer": ["Java", "Python", "Git", "OOP", "SQL", "REST APIs", "Docker"],
    "Frontend Developer": ["JavaScript", "React", "CSS", "HTML", "Web Performance", "Responsive Design"],
    "Backend Developer": ["Node.js", "Express", "MongoDB", "PostgreSQL", "Docker", "API Design"],
    "DevOps Engineer": ["Linux", "AWS", "Terraform", "CI/CD", "Docker", "Kubernetes", "Monitoring"],
}

if input_type == "Predefined Skills":
    selected_role = st.sidebar.selectbox("Select Tech Role", list(tech_roles.keys()))
    selected_skills = st.sidebar.multiselect(
        "Customize Skills", tech_roles[selected_role], default=tech_roles[selected_role]
    )
else:
    jd_text = st.sidebar.text_area("Paste Job Description Here", height=250)

# --- Submit Button ---
analyze_button = st.button("🔍 Analyze Resume")

if analyze_button:
    st.info("This is a UI demo. Backend logic is not executed.")
    st.success("✅ Frontend working perfectly. Backend is turned off for safe preview.")

    st.subheader("📊 Sample Output Preview")
    st.markdown("**Overall Score:** 85%")
    st.markdown("**Candidate Selected:** ✅ Yes")
    st.markdown("**Strengths:** Python, SQL, Pandas")
    st.markdown("**Missing Skills:** TensorFlow, MLOps")
    st.markdown("---")

    st.subheader("⚠️ Sample Weakness")
    st.markdown("**Skill:** TensorFlow — Score: 4/10")
    st.markdown("**Weakness:** Lacks clear demonstration of deep learning experience.")
    st.markdown("**Suggestions:**")
    st.markdown("- Add TensorFlow project using CNNs.")
    st.markdown("- Mention TensorBoard experience.")
    st.markdown("- Include training pipeline with TensorFlow.")
    st.markdown("**Example Addition:** Built and deployed TensorFlow-based image classifier with 90% accuracy.")

