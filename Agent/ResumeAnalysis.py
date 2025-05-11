import pypdf
import langchain
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader
import io
from langchain_groq import ChatGroq
from langchain.vectorstores import FAISS
from langchain.chains import RetrievalQA  # Fixed spelling
from concurrent.futures import ThreadPoolExecutor
from langchain_community.embeddings import HuggingFaceEmbeddings
import re
import base64
from typing import Dict, Any
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
            model="gemma2-9b-it",
            temperature=0.6, 
            streaming=True,
            verbose=True,
            api_key=GROQ_API_KEY)
        self.embedding = HuggingFaceEmbeddings()
    def safe_api_call(self, prompt: str, max_retries: int = 3) -> Dict[str, Any]:
        """Handle API calls with retries and proper error checking"""
        for attempt in range(max_retries):
            try:
                response = self.Model.invoke(prompt)
                
                # Check if response has valid content
                if not hasattr(response, 'content'):
                    raise ValueError("Invalid API response - missing content")
                
                # Try to parse JSON if content looks like JSON
                content = response.content.strip()
                if content.startswith('{') or content.startswith('['):
                    try:
                        return json.loads(content)
                    except json.JSONDecodeError:
                        pass
                
                return {"response": content}
                
            except Exception as e:
                if attempt == max_retries - 1:
                    raise
                time.sleep(1)  # Wait before retrying
        return {}
    
    def pdfFile(self, pdf_file):
        """Extract text from PDF file"""
        try:
            if hasattr(pdf_file, 'getvalue'):
                pdf_data = pdf_file.getvalue()
                pdf_file_like = io.BytesIO(pdf_data)
                loader = PyPDFLoader(pdf_file_like)
            else:
                loader = PyPDFLoader(file_path=pdf_file)  # Fixed variable name
            text = "" 
            pages = loader.load_and_split()
            for page in pages:
                text += page.page_content + "\n"
            return text
        except Exception as e:
            print(f"Error processing PDF: {e}")
            return None

    def txtFile(self, text_file):
        """Extract text from text file"""
        try:
            if hasattr(text_file, "getvalue"):
                return text_file.getvalue().decode("utf-8") 
            else:
                with open(text_file, "r", encoding="utf-8") as f:  # Added encoding
                    return f.read()
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
            
    def rag_vector_store(self, text, chunk_size=700, chunk_overlap=200):
        """Create vector store from text"""
        if not text:
            raise ValueError("No text provided for vector store creation")
            
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap
        )
        chunks = text_splitter.split_text(text)
        vector_store = FAISS.from_texts(chunks, embedding=self.embedding)  # Fixed chunks usage
        return vector_store
    
    def qa_chain(self, text=None):
        """Create QA chain for question answering"""
        if text:
            vectorstore = self.rag_vector_store(text)
        elif self.rag_vectorstore:
            vectorstore = self.rag_vectorstore
        else:
            raise ValueError("No text provided for QA chain creation")
            
        retriever = vectorstore.as_retriever(search_kwargs={"k":1})
        chain = RetrievalQA.from_chain_type(
            llm=self.Model,
            retriever=retriever,
            chain_type="stuff"
        )
        return chain
    
    def analyze_Skills(self, qa_chain, skills):
        """Improved skill analysis with better error handling"""
        query = f"On a scale of 0-10, how clearly does the person demonstrate proficiency in {skills}? Provide the numeric rating before the reasoning."
        
        try:
            response = qa_chain.run(query)
            if not response:
                return skills, 0, "No response from model"
                
            match = re.search(r"(\d{1,2})", response)
            score = int(match.group(1)) if match else 0
            
            reasoning = response.split(".", 1)[1].strip() if "." in response and len(response.split(".")) > 1 else ""
            return skills, min(score, 10), reasoning
            
        except Exception as e:
            print(f"Error analyzing skills: {e}")
            return skills, 0, f"Error in analysis: {str(e)}"
    
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
            {self.resume_text[:3000]}...
           
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
        
        self.resume_weakness = weaknesses
        return weaknesses
    
    def extract_skills_from_job_description(self, jd_text):
        """Extract skills from job description"""
        try:
            prompt = f"""
            Extract a comprehensive list of skills, technologies and competencies required from this
            job description.
            Format the output as a Python list of strings, only include the list nothing else.
            
            Job description:
            {jd_text}
            """
            response = self.Model.invoke(prompt)
            skills_text = response.content

            match = re.search(r'\[(.*?)\]', skills_text, re.DOTALL)

            if match:
                skills_text = match.group(0)

            try:
                skills_list = eval(skills_text)
                if isinstance(skills_list, list):
                    return skills_list
            except:
                pass
                
            skills = []
            for line in skills_text.split("\n"):
                line = line.strip()
                if line.startswith('-') or line.startswith('*'):
                    skill = line[2:].strip()
                    if skill:
                        skills.append(skill)
                elif line.startswith('"') and line.endswith('"'):
                    skill = line.strip('"')
                    if skill:
                        skills.append(skill)
            return skills
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
            # Create vector store and QA chain
            vectorstore = self.rag_vector_store(text=resume_text)
            retriever = vectorstore.as_retriever(search_kwargs={"k": 3})  # Increase retrieved chunks
            qa_chain = RetrievalQA.from_chain_type(
                llm=self.Model,
                retriever=retriever,
                chain_type="stuff",
                return_source_documents=False
            )

            # Analyze skills in parallel
            with ThreadPoolExecutor(max_workers=min(5, len(skills))) as executor:
                results = list(executor.map(lambda skill: self.analyze_Skills(qa_chain, skill), skills))

            # Process results
            skills_score = {}
            skills_reasoning = {}
            total_score = 0
            valid_skills_count = 0

            for skill, score, reasoning in results:
                if skill and score is not None:
                    score = min(max(score, 0), 10)  # Clamp score between 0-10
                    skills_score[skill] = score
                    skills_reasoning[skill] = reasoning or "No reasoning provided"
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

            # Improved weakness detection
            improvement_areas = []
            if overall_score < 70:  # More aggressive weakness detection for lower scores
                improvement_areas.extend(weak_skills)
                if overall_score < 50:  # Include medium skills if score is very low
                    improvement_areas.extend(medium_skills)

            # Ensure we always show some weaknesses for scores below cutoff
            if not improvement_areas and overall_score < self.cutoff_score:
                improvement_areas = [skill for skill, score in skills_score.items() 
                                if score < max(7, np.percentile(list(skills_score.values()), 70))]

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
                "improvement_areas": list(set(improvement_areas)),  # Remove duplicates
                "medium_skills": medium_skills  # New field for borderline skills
            }

            return self.analyze_result

        except Exception as e:
            error_msg = f"Analysis failed: {str(e)}"
            if hasattr(self, 'last_error'):
                self.last_error = error_msg
            raise ValueError(error_msg)

    def chat_with_resume(self, query):
        """Chat with resume content"""
        try:
            if not self.resume_text:
                raise ValueError("No resume text loaded. Upload your resume first.")
            if not self.rag_vectorstore:
                self.rag_vectorstore = self.rag_vector_store(self.resume_text)
                
            retriever = self.rag_vectorstore.as_retriever()
            chain = RetrievalQA.from_chain_type(
                llm=self.Model,
                retriever=retriever,
                chain_type="stuff",
                return_source_documents=False
            )
            response = chain.invoke({"query": query})
            return response["result"]
        except Exception as e:
            print(f"Error in chat_with_resume: {e}")
            return f"Error: {str(e)}"

    def generate_question_answer(self, interview_type: str, difficulty: str, num_questions: int):
        prompt = f"""
        Generate exactly {num_questions} interview questions for {interview_type} ({difficulty} level).
        Return ONLY a JSON array with each item containing:
        {{
            "question": "text",
            "answer": "text",
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
            
            return questions[:num_questions]  # Ensure exact count
            
        except Exception as e:
            self.last_error = str(e)
            # Return simple fallback questions
            return [{
                "question": f"{interview_type} interview question ({difficulty})",
                "answer": "Sample answer based on resume",
                "category": "general"
            } for _ in range(num_questions)]
        
    def generate_tailored_documents(self, role: str) -> tuple[str, str]:
        try:
            # Build a more structured prompt
            prompt = f"""
            Generate tailored resume and cover letter for: {role}
            
            Requirements:
            1. RESUME FORMAT:
            - Professional summary
            - Technical skills section
            - Work experience with metrics
            - Education
            
            2. COVER LETTER FORMAT:
            - 3-4 paragraphs
            - Address hiring manager
            - Highlight relevant skills
            - Show enthusiasm
            
            3. FOCUS AREAS:
            - {', '.join(self.resume_strengths[:3]) if hasattr(self, 'resume_strengths') else 'Relevant skills'}
            
            Return as valid JSON with 'resume' and 'cover_letter' keys.
            """
            
            response = self.Model.invoke(prompt)
            content = response.content.strip()
            
            # Try parsing as JSON first
            try:
                docs = json.loads(content)
                resume = docs.get('resume', '').strip()
                cover = docs.get('cover_letter', '').strip()
            except json.JSONDecodeError:
                # Fallback to text parsing
                if "RESUME:" in content and "COVER LETTER:" in content:
                    parts = content.split("COVER LETTER:")
                    resume = parts[0].replace("RESUME:", "").strip()
                    cover = parts[1].strip()
                else:
                    # Final fallback - split by paragraphs
                    parts = content.split("\n\n")
                    resume = "\n\n".join(parts[:len(parts)//2])
                    cover = "\n\n".join(parts[len(parts)//2:])
            
            # Ensure minimum content length
            if len(resume) < 50 or len(cover) < 50:
                raise ValueError("Generated content too short")
                
            return resume, cover
            
        except Exception as e:
            self.last_error = str(e)
            # Return template fallback
            return (
                "Professional Resume Content\n\nSkills: [...]\nExperience: [...]",
                "Dear Hiring Manager,\n\nI'm excited to apply...\n\nSincerely,\n[Your Name]"
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