import pypdf
import PyPDF2
import pdfplumber  # Additional PDF library as fallback
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader
from langchain_huggingface import HuggingFaceEmbeddings
import io
from langchain_groq import ChatGroq
from langchain_community.vectorstores import FAISS
from concurrent.futures import ThreadPoolExecutor
import re
import json
import warnings
import os
import numpy as np
import time
from dotenv import load_dotenv

warnings.filterwarnings("ignore")
load_dotenv()

GROQ_API_KEY = os.environ.get("GROQ_API_KEY")

class Resume_Analysis:
    
    def __init__(self, cutoff_score=75):
        self.cutoff_score = cutoff_score
        self.resume_text = None
        self.rag_vectorstore = None
        self.analyze_result = None
        self.jd_text = None
        self.extracted_skills = None
        self.resume_weakness = None
        self.resume_strengths = None
        self.improvement_suggestion = None
        self.Model = ChatGroq( 
            model="openai/gpt-oss-20b",
            temperature=0.6, 
            api_key=GROQ_API_KEY)
        
        # Use FREE local embeddings instead of Google Gemini
        self.embedding = HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-MiniLM-L6-v2",
            model_kwargs={'device': 'cpu'}
        )
    
    def pdfFile_pypdf(self, pdf_file):
        """Extract text using pypdf (primary method)"""
        try:
            if hasattr(pdf_file, 'getvalue'):
                pdf_data = pdf_file.getvalue()
                pdf_reader = pypdf.PdfReader(io.BytesIO(pdf_data))
            else:
                pdf_reader = pypdf.PdfReader(pdf_file)
            
            text = ""
            for page in pdf_reader.pages:
                text += page.extract_text() + "\n"
            return text.strip()
        except Exception as e:
            print(f"pypdf failed: {e}")
            return None

    def pdfFile_pypdf2(self, pdf_file):
        """Extract text using PyPDF2 (fallback method)"""
        try:
            if hasattr(pdf_file, 'getvalue'):
                pdf_data = pdf_file.getvalue()
                pdf_reader = PyPDF2.PdfReader(io.BytesIO(pdf_data))
            else:
                pdf_reader = PyPDF2.PdfReader(pdf_file)
            
            text = ""
            for page in pdf_reader.pages:
                text += page.extract_text() + "\n"
            return text.strip()
        except Exception as e:
            print(f"PyPDF2 failed: {e}")
            return None

    def pdfFile_pdfplumber(self, pdf_file):
        """Extract text using pdfplumber (robust fallback)"""
        try:
            if hasattr(pdf_file, 'getvalue'):
                pdf_data = pdf_file.getvalue()
                pdf_file_like = io.BytesIO(pdf_data)
            else:
                pdf_file_like = pdf_file
            
            text = ""
            with pdfplumber.open(pdf_file_like) as pdf:
                for page in pdf.pages:
                    page_text = page.extract_text()
                    if page_text:
                        text += page_text + "\n"
            return text.strip()
        except Exception as e:
            print(f"pdfplumber failed: {e}")
            return None

    def pdfFile_langchain(self, pdf_file):
        """Extract text using LangChain's PyPDFLoader"""
        try:
            if hasattr(pdf_file, 'getvalue'):
                pdf_data = pdf_file.getvalue()
                pdf_file_like = io.BytesIO(pdf_data)
                loader = PyPDFLoader(pdf_file_like)
            else:
                loader = PyPDFLoader(file_path=pdf_file)
            
            text = ""
            pages = loader.load_and_split()
            for page in pages:
                text += page.page_content + "\n"
            return text.strip()
        except Exception as e:
            print(f"LangChain PDF loader failed: {e}")
            return None

    def pdfFile(self, pdf_file):
        """Extract text from PDF file with multiple fallback methods"""
        methods = [
            self.pdfFile_pdfplumber,  # Most robust for corrupted PDFs
            self.pdfFile_pypdf,       # Primary method
            self.pdfFile_pypdf2,      # Alternative method
            self.pdfFile_langchain    # LangChain's method
        ]
        
        for i, method in enumerate(methods):
            try:
                print(f"Trying PDF extraction method {i+1}...")
                text = method(pdf_file)
                if text and len(text.strip()) > 50:  # Ensure we have meaningful text
                    print(f"Success with method {i+1}")
                    return text
            except Exception as e:
                print(f"Method {i+1} failed: {e}")
                continue
        
        print("All PDF extraction methods failed")
        return None

    def txtFile(self, text_file):
        """Extract text from text file"""
        try:
            if hasattr(text_file, "getvalue"):
                content = text_file.getvalue()
                if isinstance(content, bytes):
                    return content.decode("utf-8")
                return content
            else:
                with open(text_file, "r", encoding="utf-8") as f:
                    return f.read()
        except UnicodeDecodeError:
            # Try different encodings
            try:
                if hasattr(text_file, "getvalue"):
                    return text_file.getvalue().decode("latin-1")
                else:
                    with open(text_file, "r", encoding="latin-1") as f:
                        return f.read()
            except Exception as e:
                print(f"Error reading text file with latin-1: {e}")
                return None
        except Exception as e:
            print(f"Error reading text file: {e}")
            return None

    def FileSelection(self, file):
        """Select appropriate file handler based on file extension"""
        try:
            if hasattr(file, "name"):
                file_extension = file.name.split(".")[-1].lower()
            else:
                file_extension = file.split(".")[-1].lower()

            if file_extension == "pdf":
                return self.pdfFile(file)
            elif file_extension == "txt":
                return self.txtFile(file)  
            else:
                print(f"Unsupported file extension: {file_extension}")
                return None
        except Exception as e:
            print(f"File selection error: {e}")
            return None
            
    def validate_text_content(self, text):
        """Validate that extracted text is usable"""
        if not text:
            return False
        if len(text.strip()) < 50:  # Too short to be a resume
            return False
        if text.strip().count('\n') < 3:  # Not enough structure
            return False
        return True
            
    def rag_vector_store(self, text, chunk_size=700, chunk_overlap=200):
        """Create vector store from text using FREE embeddings"""
        if not text:
            raise ValueError("No text provided for vector store creation")
            
        if not self.validate_text_content(text):
            raise ValueError("Extracted text is too short or invalid for analysis")
            
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap
        )
        chunks = text_splitter.split_text(text)
        
        if not chunks:
            raise ValueError("No chunks created from the text")
            
        vector_store = FAISS.from_texts(chunks, embedding=self.embedding)
        return vector_store
    
    def create_simple_qa_chain(self, text=None):
        """Create a simple QA chain"""
        if text:
            vectorstore = self.rag_vector_store(text)
        elif self.rag_vectorstore:
            vectorstore = self.rag_vectorstore
        else:
            raise ValueError("No text provided for QA chain creation")
            
        retriever = vectorstore.as_retriever(search_kwargs={"k": 3})
        return retriever
    
    def analyze_Skills(self, retriever, skill):
        """Improved skill analysis with better error handling"""
        query = f"How clearly does the resume demonstrate proficiency in {skill}? Rate from 0-10 and explain."
        
        try:
            # Get relevant documents
            docs = retriever.get_relevant_documents(query)
            if not docs:
                return skill, 0, "No relevant content found in resume"
                
            context = "\n".join([doc.page_content for doc in docs][:3])
            
            # Create prompt with context
            prompt = f"""
            Based on this resume content:
            {context}
            
            Question: {query}
            
            Provide your answer in this exact format:
            Score: [number between 0-10]
            Reasoning: [your explanation here]
            """
            
            response = self.Model.invoke(prompt)
            if not response:
                return skill, 0, "No response from model"
                
            answer = response.content
            
            # Extract score using regex
            score_match = re.search(r"Score:\s*(\d{1,2})", answer, re.IGNORECASE)
            if not score_match:
                # Fallback: look for any number
                score_match = re.search(r"\b(\d{1,2})\b", answer.split('\n')[0])
            
            score = int(score_match.group(1)) if score_match else 0
            score = min(max(score, 0), 10)  # Ensure score is between 0-10
            
            # Extract reasoning
            reasoning_match = re.search(r"Reasoning:\s*(.+)", answer, re.IGNORECASE | re.DOTALL)
            reasoning = reasoning_match.group(1).strip() if reasoning_match else answer
                
            return skill, score, reasoning
            
        except Exception as e:
            print(f"Error analyzing {skill}: {e}")
            return skill, 0, f"Error in analysis: {str(e)}"
    
    def analyze_resume_weakness(self):
        """Analyze resume weaknesses"""
        if not self.extracted_skills or not self.resume_text or not self.analyze_result:
            return []
        
        weaknesses = []

        for skill in self.analyze_result.get("missing_skills", []):
            prompt = f"""
            Analyze why the resume is weak regarding {skill}.

            For your analysis consider:
            1: What is missing from the resume regarding this skill?
            2: How could it be improved with specific examples?
            3: What specific action items would make this skill stand out?

            Resume content:
            {self.resume_text[:3000]}
           
            Provide your response in JSON format:
            {{
                "weakness": "A concise description of weakness (1-2 sentences)",
                "improvement_suggestion": [
                    "specific suggestion 1",
                    "specific suggestion 2", 
                    "specific suggestion 3"
                ],
                "example_addition": "A specific bullet point that could be added to showcase the skill"
            }}
            Return only valid JSON with no additional text.
            """

            try:
                response = self.Model.invoke(prompt)
                weakness_content = response.content.strip()

                try:
                    weakness_data = json.loads(weakness_content)

                    weakness_detail = {
                        "skill": skill,
                        "score": self.analyze_result.get("skills_score", {}).get(skill, 0),
                        "detail": weakness_data.get("weakness", "No specific detail provided."),
                        "suggestion": weakness_data.get("improvement_suggestion", []),
                        "example": weakness_data.get("example_addition", "")
                    }
                    weaknesses.append(weakness_detail)

                except json.JSONDecodeError:
                    weaknesses.append({
                        "skill": skill,
                        "score": self.analyze_result.get("skills_score", {}).get(skill, 0),
                        "detail": weakness_content[:200]
                    })
            except Exception as e:
                print(f"Error analyzing weakness for {skill}: {e}")
                weaknesses.append({
                    "skill": skill,
                    "score": self.analyze_result.get("skills_score", {}).get(skill, 0),
                    "detail": f"Error analyzing this skill: {str(e)}"
                })
        
        self.resume_weakness = weaknesses
        return weaknesses
    
    def extract_skills_from_job_description(self, jd_text):
        """Extract skills from job description"""
        try:
            prompt = f"""
            Extract a comprehensive list of technical skills, technologies and competencies from this job description.
            Return ONLY a Python list format like: ["skill1", "skill2", "skill3"]
            
            Job description:
            {jd_text[:2000]}
            """
            response = self.Model.invoke(prompt)
            skills_text = response.content

            # Extract list from response using regex
            match = re.search(r'\[.*\]', skills_text, re.DOTALL)
            if match:
                try:
                    skills_list = eval(match.group(0))
                    if isinstance(skills_list, list):
                        # Clean and return skills
                        return [str(skill).strip() for skill in skills_list if skill and str(skill).strip()]
                except:
                    pass
            
            # Fallback: extract skills line by line
            skills = []
            for line in skills_text.split("\n"):
                line = line.strip().strip('",')
                if line and len(line) > 2 and not line.startswith('[') and not line.startswith(']'):
                    skills.append(line)
                    
            return skills[:20]  # Limit to 20 skills max
            
        except Exception as e:
            print(f"Error extracting skills from job description: {e}")
            return []

    def semantic_skills_analysis(self, resume_text, skills):
        """Analyze skills semantically with improved weakness detection and scoring"""
        if not resume_text:
            raise ValueError("No resume text provided for analysis")
        if not skills:
            raise ValueError("No skills provided for analysis")

        try:
            # Create vector store and retriever
            vectorstore = self.rag_vector_store(text=resume_text)
            retriever = vectorstore.as_retriever(search_kwargs={"k": 3})

            # Analyze skills in parallel
            with ThreadPoolExecutor(max_workers=min(3, len(skills))) as executor:
                results = list(executor.map(lambda skill: self.analyze_Skills(retriever, skill), skills))

            # Process results
            skills_score = {}
            skills_reasoning = {}
            total_score = 0
            valid_skills_count = 0

            for skill, score, reasoning in results:
                if skill and score is not None:
                    skills_score[skill] = score
                    skills_reasoning[skill] = reasoning
                    total_score += score
                    valid_skills_count += 1

            # Calculate overall score
            overall_score = 0
            if valid_skills_count > 0:
                overall_score = int((total_score / (10 * valid_skills_count)) * 100)
            
            # Determine strengths and weaknesses
            strengths = [skill for skill, score in skills_score.items() if score >= 7]
            weak_skills = [skill for skill, score in skills_score.items() if score <= 5]
            medium_skills = [skill for skill, score in skills_score.items() if 5 < score < 7]

            # Prepare final result
            self.resume_strengths = strengths
            self.analyze_result = {
                "overall_score": overall_score,
                "skills_score": skills_score,
                "skills_reasoning": skills_reasoning,
                "selected": overall_score >= self.cutoff_score,
                "reasoning": f"Evaluation based on {valid_skills_count} analyzed skills",
                "missing_skills": weak_skills,
                "strengths": strengths,
                "improvement_areas": weak_skills,
                "medium_skills": medium_skills
            }

            return self.analyze_result

        except Exception as e:
            error_msg = f"Analysis failed: {str(e)}"
            raise ValueError(error_msg)

    def chat_with_resume(self, query):
        """Chat with resume content using simple retrieval"""
        try:
            if not self.resume_text:
                raise ValueError("No resume text loaded. Upload your resume first.")
            if not self.rag_vectorstore:
                self.rag_vectorstore = self.rag_vector_store(self.resume_text)
                
            # Get relevant documents
            retriever = self.rag_vectorstore.as_retriever(search_kwargs={"k": 3})
            docs = retriever.get_relevant_documents(query)
            context = "\n".join([doc.page_content for doc in docs])
            
            # Create prompt with context
            prompt = f"""
            Based on the following resume content:
            {context}
            
            Question: {query}
            
            Provide a helpful and accurate answer:
            """
            
            response = self.Model.invoke(prompt)
            return response.content
            
        except Exception as e:
            print(f"Error in chat_with_resume: {e}")
            return f"Error: {str(e)}"

    def generate_question_answer(self, interview_type: str, difficulty: str, num_questions: int):
        prompt = f"""
        Generate exactly {num_questions} interview questions for {interview_type} position at {difficulty} level.
        Return ONLY a JSON array with each item containing:
        {{
            "question": "the interview question",
            "answer": "a good sample answer", 
            "category": "technical/behavioral/general"
        }}
        No additional commentary or formatting.
        """
        
        try:
            response = self.Model.invoke(prompt)
            content = response.content.strip()
            
            # Clean response
            content = content.replace("```json", "").replace("```", "").strip()
            questions = json.loads(content)
            
            return questions[:num_questions]
            
        except Exception as e:
            print(f"Error generating questions: {e}")
            # Return simple fallback questions
            return [{
                "question": f"{interview_type} question {i+1} ({difficulty})",
                "answer": "This would be a sample answer based on the resume content.",
                "category": "general"
            } for i in range(num_questions)]
        
    def generate_tailored_documents(self, role: str):
        try:
            # Build a more structured prompt
            prompt = f"""
            Generate tailored resume content and cover letter for the role: {role}
            
            Focus on these strengths: {', '.join(self.resume_strengths[:3]) if self.resume_strengths else 'relevant technical skills'}
            
            Return as valid JSON with exactly these two keys:
            {{
                "resume": "professional resume content here",
                "cover_letter": "professional cover letter here" 
            }}
            """
            
            response = self.Model.invoke(prompt)
            content = response.content.strip()
            
            # Try parsing as JSON first
            try:
                docs = json.loads(content)
                resume = docs.get('resume', '').strip()
                cover = docs.get('cover_letter', '').strip()
            except json.JSONDecodeError:
                # Fallback
                resume = f"Professional Resume for {role}\n\nHighlighting: {', '.join(self.resume_strengths[:3]) if self.resume_strengths else 'relevant skills'}"
                cover = f"Dear Hiring Manager,\n\nI am excited to apply for the {role} position...\n\nSincerely,\n[Your Name]"
            
            return resume, cover
            
        except Exception as e:
            print(f"Error generating documents: {e}")
            return (
                f"Professional Resume for {role}",
                f"Cover Letter for {role} Position"
            )
        
    def reset(self):
        """Reset all class attributes"""
        self.resume_text = None
        self.rag_vectorstore = None
        self.analyze_result = None
        self.jd_text = None
        self.extracted_skills = None
        self.resume_weakness = None
        self.resume_strengths = None
        self.improvement_suggestion = None