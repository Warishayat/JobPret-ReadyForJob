import pypdf
import langchain
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_ollama import ChatOllama 
from langchain_ollama import OllamaEmbeddings
from langchain_community.document_loaders import PyPDFLoader
import io
from langchain.vectorstores import FAISS
from langchain.chains import RetrivalQA
from concurrent.futures import ThreadPoolExecutor
import re
import json
import warnings
warnings.filterwarnings("ignore")
class Resume_Analysis:
    
    def __init__(self,cutoff_score=75):
        self.cutoff_score = cutoff_score
        self.resume_text = None
        self.rag_vectorstore = None
        self.analyze_result = None
        self.jd_text =  None
        self.extracted_skills = None
        self.resume_weakness = None
        self.resume_strengths = None
        self.inprovement_suggestion = None

    #if user pass pdf
    def pdfFile(self,pdf_file):
        "extracted text from the pdf"
        try:
            if hasattr(pdf_file,'getvalue'):
                pdf_data = pdf_file.getvalue()
                pdf_file_like = io.BytesIO(pdf_data)
                loader = PyPDFLoader(pdf_file_like)
            else:
                loader = PyPDFLoader(file_path=pdf_data)
            text = "" 
            pages = loader.load_and_split()
            for page in pages:
                text += page.page_content + "\n"
            return text
        except Exception as e:
            print(f"Issue happen: {e}")

    #if user pass text file
    def txtFile(self,text_file):
        try:
            if hasattr(text_file,"getvalue"):
                return text_file.getvalue().decode("utf-8") 
            else:
                with open(text_file , "rb") as f:
                    return f.read()
        except Exception as e:
            print(f"you may have some issue at:{e}")

    #Selection of pdf or text file
    def FileSelection(self,file):
        try:
            if hasattr(file,"name"):
               file_extention = file.name.split(".")[-1].lower()
            else:
                file_extention = file.split(".")[-1].lower()

            if file_extention =="pdf":
                return self.pdfFile(file)
            elif file_extention == "txt":
                return self.txtFile(file)  
            else:
                print(f"Unsupported file extension: {file_extention}")    
        except Exception as e:
            print(f"You have some issue at {e}")
            
    def rag_vector_store(self,text,chunk_size=700,chunk_overlap=200):
        text_splitter = RecursiveCharacterTextSplitter(chunk_size,chunk_overlap)
        chunks = text_splitter.split_text(text)

        #load embeddings
        embeddings = OllamaEmbeddings(model="nomic-embed-text")
        vector_store = FAISS.from_texts([chunks],embedding=embeddings)
        return vector_store
    
    # #qreating chain function
    def qa_chain(self):
        Model = ChatOllama(
        model="llama3.2:1b",
            temperature=0.7,
            verbose=True
        )

        vectorstore=self.rag_vector_store()  #will pass text here so i could get the vectorstore point to remmber when run
        retriver=vectorstore.as_retriever(kwargs={"k":1})
        chain = RetrivalQA.from_chain_type(
            llm=Model,
            retriever = retriver,
            chain_type = "stuff"
        )
        return chain
    
    #analyze the skills based on resume
    #analyze the skills and grade them.
    def analyze_Skills(self,qa_chain,skills):
        "analyze the ksills from resume"

        query = f"on the scale of 0-10,how clearly does the person proficency in the {skills}?provide the numerice rating before the reasoning."
        response = qa_chain.run(query)
        match = re.search(r"(\d{1,2})",response)
        score = int(match.group(1)) if match else 0
        
        reasoning = response.split(".",1)[1].strip() if "." in response and len(response.split(".")) > 1 else ""

        return skills,min(score,10), reasoning
    
    #analyze resume weekness
    def analyze_resume_weekness(self):
        "analyze the resume based on the missing skills"
        if not self.extracted_skills or not self.resume_text or not self.analyze_result:
            return []
        
        weakness = []

        #intialize the model
        Model = ChatOllama(
            model="llama3.2:1b",
                temperature=0.7,
                verbose=True
            )

        for skills in self.analyze_result.get("missing_skills",[]):
            

            prompt = f"""
            analyze why the resume is week and demonstarte the {skills}.

            for you analysis considerd:
            1: What is missing from the resume regarding the skills?
            2: How could it be removed with specif example?
            3: what specific action items would make the skills standout?

            resume content:
           {self.resume_text[:3000]}...
           
           provide your response in the json formate:
           {{
                "weakness" : "A Concise discription of weakness what is missing or problem in (1-2 sentense.),
                "improvement_suggestion": [
                "specific suggestion 1",
                "specific suggestion 2",
                "specific suggestion 3",
                ],
                "example_addition" : A specific bullet point that could be added to shoecase the skills.
           }}
           Return only valid json no any other formate text.
            """

        response = Model.invoke(prompt)
        weakness_content = response.content.strip()

        try:
            weakness_data = json.loads(weakness_content)

            weakness_detail = {
                "skill " : skills,
                "score" : self.analyze_result.get("skills_score",{}).get(skills,0),
                "detail" : weakness_data.get("weakness","No specific detail provided."),
                "suggestion" : weakness_data.get("improvement_suggestion",[]),
                "example" : weakness_data.get("example_added","")
            }
            weakness.append(weakness_detail)

        except json.JSONDecodeError:
            weakness.append({
                "skills": skills,
                "score": self.analyze_result.get("skills_score",{}).get(skills,0),
                "detail" : weakness_content[:200]
            })
        self.resume_weakness = weakness
        return weakness
    
    #incase user upload job discription
    def extracted_skills_from_job_discription(self,jd_text):
        "extarcted skills from the job discription"
        Model = ChatOllama(
            model="llama3.2:1b",
                temperature=0.7,
                verbose=True
            )
        
        try:
            prompt = f"""
            extacrt the comprehensive list of skills technologies and compentencies required from this
            job discription.
            formate the output as a python list of strings,only include the list nothing else.
            
            job discription:
            {jd_text}
            """
            response = Model.invoke(prompt)
            skills_text = response.content

            match = re.search(r'\[(.*?)\]',skills_text,re.DOTALL)

            if match:
                skills_text = match.group(0)

            try:
                skills_list = eval(skills_text)
                if isinstance(skills_list,list):
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
           print(f"Error from extarcting skills from the job discription: {e}")

    #sematic skills analysis:
    def semantic_skills_analysis(self,resume_text,skills):
        "analyze the skills sematically."
        Model = ChatOllama(
            model="llama3.2:1b",
                temperature=0.7,
                verbose=True
            )
        vectorstore = self.rag_vector_store(text=resume_text)
        retriever = vectorstore.as_retriever()
        qa_chain = RetrivalQA.from_chain_type(
            llm = Model,
            retriever = retriever,
            chain_type = "stuff",
            return_source_documents = False
        )
        skills_score = {}
        skills_reasoning = {}
        missing_skills = {}
        total_score = 0

        with ThreadPoolExecutor(max_workers=5) as executor:
            result = list(executor.map(lambda skill:self.analyze_Skills(qa_chain,skill),skills))

        for skill,score,reasoning in result:
            skills_score[skill] = score
            skills_reasoning[skill] = reasoning
            total_score += score

            if score <= 5:
                missing_skills.append(skill)
                    
        overall_score = int((total_score/(10*len(skills)))*100)
        selected = overall_score >= self.cutoff_score

        reasoning = "candidate based on explicit resume content using semantic similarity and clear numeric scoring."
        strengths = [skills for skill,score in skills_score.items() if score >=7]
        improvement_areas = missing_skills if not selected else []

        self.resume_strengths = strengths

        return {
            "overall_score" : overall_score,
            "skills_score"  : skills_score,
            "skills_reasoning" : skills_reasoning , 
            "selected" : selected ,
            "reasoning" : reasoning ,
            "missing_skills" : missing_skills,
            "strength" : strengths,
            "improvement_areas" : improvement_areas

        }

