import pypdf
import PyPDF2
import pdfplumber
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
            model="llama-3.1-8b-instant",
            temperature=0.3,
            api_key=GROQ_API_KEY)
        
        self.embedding = HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-MiniLM-L6-v2",
            model_kwargs={'device': 'cpu'}
        )
    
    def pdfFile(self, pdf_file):
        """Fast PDF extraction"""
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
            return text.strip() if text else None
        except Exception as e:
            print(f"PDF extraction failed: {e}")
            return None

    def txtFile(self, text_file):
        """Fast text file extraction"""
        try:
            if hasattr(text_file, "getvalue"):
                content = text_file.getvalue()
                return content.decode("utf-8") if isinstance(content, bytes) else content
            else:
                with open(text_file, "r", encoding="utf-8") as f:
                    return f.read()
        except Exception as e:
            print(f"Text file error: {e}")
            return None

    def FileSelection(self, file):
        """Fast file selection"""
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
                print(f"Unsupported file type: {file_extension}")
                return None
        except Exception as e:
            print(f"File error: {e}")
            return None
            
    def rag_vector_store(self, text, chunk_size=800, chunk_overlap=100):
        """Fast vector store creation"""
        if not text or len(text.strip()) < 50:
            raise ValueError("Invalid text for analysis")
            
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap
        )
        chunks = text_splitter.split_text(text)
        
        if not chunks:
            raise ValueError("No chunks created")
            
        return FAISS.from_texts(chunks, embedding=self.embedding)
    
    def create_retriever(self, text=None):
        """Create retriever with correct method names"""
        if text:
            vectorstore = self.rag_vector_store(text)
        elif self.rag_vectorstore:
            vectorstore = self.rag_vectorstore
        else:
            raise ValueError("No text available")
            
        return vectorstore.as_retriever(search_kwargs={"k": 3})
    
    def get_relevant_documents(self, retriever, query):
        """Universal method to get documents from retriever - handles different LangChain versions"""
        try:
            # Try the new method first (LangChain >=0.1.0)
            if hasattr(retriever, 'invoke'):
                docs = retriever.invoke(query)
                return docs
            # Try the old method (LangChain <0.1.0)
            elif hasattr(retriever, 'get_relevant_documents'):
                return retriever.get_relevant_documents(query)
            else:
                # Fallback: try direct access
                return retriever(query)
        except Exception as e:
            print(f"Error getting documents: {e}")
            return []
    
    def batch_analyze_skills(self, skills, retriever, resume_text):
        """Analyze multiple skills in a single API call"""
        # Get context once for all skills
        context_queries = ["skills", "experience", "technologies", "projects"]
        all_context = ""
        
        for query in context_queries:
            try:
                docs = self.get_relevant_documents(retriever, query)
                if docs:
                    for doc in docs:
                        content = doc.page_content if hasattr(doc, 'page_content') else str(doc)
                        if content not in all_context:
                            all_context += content + "\n"
            except Exception as e:
                print(f"Context query failed for {query}: {e}")
                continue
        
        # If no context found, use resume preview
        if not all_context.strip():
            all_context = resume_text[:1500]
        
        # Create batch prompt for all skills
        skills_list = "\n".join([f"- {skill}" for skill in skills])
        
        prompt = f"""
        Analyze the resume content and evaluate proficiency for each skill below.
        For each skill, provide a score 0-10 and brief reasoning.
        
        Resume Content:
        {all_context[:2000]}
        
        Skills to evaluate:
        {skills_list}
        
        Respond in EXACT JSON format:
        {{
            "skills": {{
                "skill_name": {{
                    "score": 0-10,
                    "reasoning": "brief explanation"
                }}
            }}
        }}
        
        Be concise and focus on evidence in the resume.
        """
        
        try:
            start_time = time.time()
            response = self.Model.invoke(prompt)
            print(f"Batch analysis took: {time.time() - start_time:.2f}s")
            
            content = response.content.strip()
            
            # Extract JSON from response
            json_match = re.search(r'\{.*\}', content, re.DOTALL)
            if json_match:
                result_data = json.loads(json_match.group())
                skills_data = result_data.get("skills", {})
                
                results = []
                for skill in skills:
                    skill_data = skills_data.get(skill, {})
                    score = skill_data.get("score", 0)
                    reasoning = skill_data.get("reasoning", "No specific evidence found")
                    results.append((skill, score, reasoning))
                
                return results
            else:
                # Fallback: return default scores
                return [(skill, 0, "Analysis failed") for skill in skills]
                
        except Exception as e:
            print(f"Batch analysis error: {e}")
            return [(skill, 0, f"Error: {str(e)}") for skill in skills]
    
    def analyze_resume_weakness(self):
        """Fast weakness analysis"""
        if not self.analyze_result:
            return []
        
        weaknesses = []
        missing_skills = self.analyze_result.get("missing_skills", [])[:3]  # Limit to 3
        
        for skill in missing_skills:
            prompt = f"""
            Quickly suggest improvements for missing skill: {skill}
            Resume excerpt: {self.resume_text[:1000]}
            
            Respond with JSON: {{"suggestion": "brief suggestion"}}
            """
            
            try:
                response = self.Model.invoke(prompt)
                content = response.content.strip()
                
                try:
                    data = json.loads(content)
                    weaknesses.append({
                        "skill": skill,
                        "suggestion": data.get("suggestion", "Add relevant experience")
                    })
                except:
                    weaknesses.append({
                        "skill": skill,
                        "suggestion": "Consider adding projects or experience with this skill"
                    })
            except:
                weaknesses.append({
                    "skill": skill,
                    "suggestion": "Develop this skill through projects or coursework"
                })
        
        return weaknesses
    
    def extract_skills_from_job_description(self, jd_text):
        """Fast skill extraction"""
        try:
            prompt = f"""
            Extract top technical skills from this job description. Return as JSON list.
            
            JD: {jd_text[:1500]}
            
            Format: {{"skills": ["skill1", "skill2", ...]}}
            Max 10 skills.
            """
            
            response = self.Model.invoke(prompt)
            content = response.content.strip()
            
            json_match = re.search(r'\{.*\}', content, re.DOTALL)
            if json_match:
                data = json.loads(json_match.group())
                return data.get("skills", [])[:10]
            else:
                # Simple keyword fallback
                skills_keywords = ["python", "java", "javascript", "sql", "aws", "docker", "react", "node", "machine learning", "ai"]
                found_skills = []
                for skill in skills_keywords:
                    if skill.lower() in jd_text.lower():
                        found_skills.append(skill)
                return found_skills[:8]
                
        except Exception as e:
            print(f"Skill extraction error: {e}")
            return ["python", "javascript", "sql"]  # Default fallback

    def semantic_skills_analysis(self, resume_text, skills):
        """Fast analysis with single API call"""
        if not resume_text or not skills:
            raise ValueError("Missing input data")
            
        print(f"Fast analyzing {len(skills)} skills...")
        start_time = time.time()

        try:
            # Create retriever once
            retriever = self.create_retriever(resume_text)
            
            # Test retriever
            test_docs = self.get_relevant_documents(retriever, "skills")
            print(f"Retriever test: found {len(test_docs)} documents")
            
            # Batch analyze all skills in ONE API call
            results = self.batch_analyze_skills(skills, retriever, resume_text)
            
            # Process results quickly
            skills_score = {}
            skills_reasoning = {}
            total_score = 0
            valid_count = 0

            for skill, score, reasoning in results:
                score = min(max(score, 0), 10)
                skills_score[skill] = score
                skills_reasoning[skill] = reasoning
                if score > 0:
                    total_score += score
                    valid_count += 1

            # Calculate overall score
            overall_score = int((total_score / (10 * len(skills))) * 100) if skills else 0
            
            # Quick categorization
            strengths = [s for s, score in skills_score.items() if score >= 6]
            weak_skills = [s for s, score in skills_score.items() if score <= 3]

            self.resume_strengths = strengths
            self.analyze_result = {
                "overall_score": overall_score,
                "skills_score": skills_score,
                "skills_reasoning": skills_reasoning,
                "selected": overall_score >= self.cutoff_score,
                "missing_skills": weak_skills,
                "strengths": strengths,
                "improvement_areas": weak_skills
            }

            total_time = time.time() - start_time
            print(f"Analysis completed in {total_time:.2f} seconds")
            print(f"Overall score: {overall_score}%")
            
            return self.analyze_result

        except Exception as e:
            print(f"Fast analysis failed: {e}")
            # Return quick fallback
            return {
                "overall_score": 50,
                "skills_score": {skill: 5 for skill in skills},
                "selected": False,
                "missing_skills": skills,
                "strengths": []
            }

    def chat_with_resume(self, query):
        """Fast chat response"""
        try:
            if not self.rag_vectorstore:
                if not self.resume_text:
                    return "Please upload a resume first"
                self.rag_vectorstore = self.rag_vector_store(self.resume_text)
                
            retriever = self.create_retriever()
            docs = self.get_relevant_documents(retriever, query)
            
            # Extract content from documents
            context_parts = []
            for doc in docs[:2]:  # Limit to 2 docs
                content = doc.page_content if hasattr(doc, 'page_content') else str(doc)
                context_parts.append(content)
            
            context = "\n".join(context_parts)
            
            prompt = f"""
            Based on: {context[:1000]}
            Question: {query}
            Short answer:
            """
            
            response = self.Model.invoke(prompt)
            return response.content[:500]  # Limit response length
            
        except Exception as e:
            return f"Error: {str(e)}"

    def generate_question_answer(self, interview_type: str, difficulty: str, num_questions: int):
        """Fast question generation"""
        prompt = f"""
        Generate {num_questions} {difficulty} {interview_type} questions.
        Return JSON: {{"questions": [{{"question": "q", "answer": "a", "category": "c"}}]}}
        """
        
        try:
            response = self.Model.invoke(prompt)
            content = response.content.strip()
            
            json_match = re.search(r'\{.*\}', content, re.DOTALL)
            if json_match:
                data = json.loads(json_match.group())
                return data.get("questions", [])[:num_questions]
            else:
                return [{
                    "question": f"{interview_type} question",
                    "answer": "Sample answer",
                    "category": "technical"
                } for _ in range(num_questions)]
                
        except Exception as e:
            print(f"Question generation error: {e}")
            return []

    def generate_tailored_documents(self, role: str):
        """Fast document generation"""
        try:
            strengths = self.resume_strengths[:2] if self.resume_strengths else ["technical skills"]
            
            prompt = f"""
            Quick resume and cover letter for {role}.
            Focus on: {', '.join(strengths)}
            JSON: {{"resume": "content", "cover_letter": "content"}}
            """
            
            response = self.Model.invoke(prompt)
            content = response.content.strip()
            
            try:
                data = json.loads(content)
                return data.get("resume", ""), data.get("cover_letter", "")
            except:
                return f"Resume for {role}", f"Cover letter for {role}"
                
        except Exception as e:
            return f"Resume for {role}", f"Cover letter for {role}"

    def reset(self):
        """Reset all attributes"""
        self.resume_text = None
        self.rag_vectorstore = None
        self.analyze_result = None
        self.jd_text = None
        self.extracted_skills = None
        self.resume_weakness = None
        self.resume_strengths = None
        self.improvement_suggestion = None