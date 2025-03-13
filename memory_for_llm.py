from langchain_community.document_loaders import PyPDFLoader, DirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

# Step1: Load raw Pdf(s)
Data_Path="data/"
def load_pdf_files(data):
    loader=DirectoryLoader(data,
                           glob='*.pdf',
                           loader_cls=PyPDFLoader)
    documents=loader.load()
    return documents

documents=load_pdf_files(data=Data_Path)
#print("length of pfd file: ",len(documents))

# Step2: Create Chunks
def create_chunks(extracted_data):
   text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
   chunks = text_splitter.split_documents(extracted_data)
   return chunks
num_chunks=create_chunks(documents)
#print(len(num_chunks))

# Step3: Create Vector Embeddings
def get_embedding_model():
    embedding_model=HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    return embedding_model
embedding_model=get_embedding_model()


# Step 4: Store embeddings in FAISS
db_path="vector_db/"
database=FAISS.from_documents(num_chunks,embedding_model)
database.save_local(db_path)