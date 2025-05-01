import pypdf
import langchain
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_ollama import ChatOllama 
from langchain_ollama import OllamaEmbeddings
from langchain_community.document_loaders import PyPDFLoader
import io
from langchain.vectorstores import FAISS

# model = ChatOllama(
#     model="llama3.2:1b",
#     temperature=0.7,
#     verbose=True
# )

# embeddings = OllamaEmbeddings(model="nomic-embed-text")
# vectors=embeddings.embed_documents("imran khan is the famous pakistani crickter and ahmad ali is famous football player from pakistan")


class Resume_Analysis:
    
    model = ChatOllama(
    model="llama3.2:1b",
    temperature=0.7,
    verbose=True
)
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
            
    def pdfPreprocessing(self,text,chunk_size=700,chunk_overlap=200):
        text_splitter = RecursiveCharacterTextSplitter(chunk_size,chunk_overlap)
        text = text_splitter.split_documents(text)

        #load embeddings
        embeddings = OllamaEmbeddings(model="nomic-embed-text")
        vector_store = FAISS.from_documents([text],embedding=embeddings)

        return vector_store