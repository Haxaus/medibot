import os
from dotenv import load_dotenv  # Add for local env support
from langchain_huggingface import HuggingFaceEndpoint
from langchain_core.prompts import PromptTemplate
from langchain.chains import RetrievalQA
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import START, MessagesState, StateGraph

# Load environment variables for local development
load_dotenv()

# Step 1: Setup LLM (Mistral with HuggingFace)
HF_TOKEN = os.getenv("HF_TOKEN")
if not HF_TOKEN:
    raise ValueError("HF_TOKEN not found in environment variables. Set it in .env or your environment.")
HUGGINGFACE_REPO_ID = "mistralai/Mistral-7B-Instruct-v0.3"

def load_llm(huggingface_repo_id):
    llm = HuggingFaceEndpoint(
        repo_id=huggingface_repo_id,
        temperature=0.5,
        task="text-generation",
        model_kwargs={"token": HF_TOKEN, "max_length": "512"}
    )
    return llm

# Step 2: Connect LLM with FAISS and Create chain
CUSTOM_PROMPT_TEMPLATE = """
Context: {context}
Question: {question}

Use the information in the context to answer the question naturally and conversationally.
If you don’t know the answer, just say so—don’t make anything up. Stick strictly to the given context and avoid small talk.
"""

def set_custom_prompt(custom_prompt_template):
    prompt = PromptTemplate(template=custom_prompt_template, input_variables=["context", "question"])
    return prompt

# Load Database
DB_PATH = "vector_db"
embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
db = FAISS.load_local(DB_PATH, embedding_model, allow_dangerous_deserialization=True)

# Create QA chain
qa_chain = RetrievalQA.from_chain_type(
    llm=load_llm(HUGGINGFACE_REPO_ID),
    chain_type="stuff",
    retriever=db.as_retriever(search_kwargs={'k': 3}),
    chain_type_kwargs={'prompt': set_custom_prompt(CUSTOM_PROMPT_TEMPLATE)}
)

# Step 3: Define the LangGraph workflow with memory
workflow = StateGraph(state_schema=MessagesState)

def call_model(state: MessagesState):
    latest_message = state["messages"][-1].content if state["messages"] else ""
    response = qa_chain.invoke({'query': latest_message})
    return {"messages": response["result"]}

workflow.add_edge(START, "model")
workflow.add_node("model", call_model)

# Add memory to the graph
memory = MemorySaver()
app = workflow.compile(checkpointer=memory)

# Step 4: Run the conversational loop with memory
config = {"configurable": {"thread_id": "1"}}
print("Welcome to the Chatbot. Type 'quit' to exit.")
while True:
    user_query = input("Write your query: ")
    if user_query.lower() == 'quit':
        print("Chatbot: Goodbye!")
        break
    
    events = app.stream({"messages": [{"role": "user", "content": user_query}]}, config=config)
    for event in events:
        print("Chatbot: ", event["model"]["messages"])