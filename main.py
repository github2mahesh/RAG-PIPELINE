from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_ollama import OllamaEmbeddings, ChatOllama
from langchain_chroma import Chroma
import os

# load the document
file_path = "C:/New Volume (D)/Python Practice/RAG/data/health.pdf"
loader = PyPDFLoader(file_path)
docs = loader.load()

# split the document into chunks
text_splitter = RecursiveCharacterTextSplitter(chunk_size=300, chunk_overlap=30)
texts = text_splitter.split_documents(docs)

# create embedding model
embeddings = OllamaEmbeddings(
    model="nomic-embed-text"
)

# convert document chunks into embeddings. this line is not necessary but just for understanding
chunk_embeddings = embeddings.embed_documents([text.page_content for text in texts])

# store in Chroma vector DB
persist_dir = "chroma_db"

if not os.path.exists(persist_dir):
    print("Creating new vector database...")
    
    vector_store = Chroma.from_documents(
        documents=texts,
        embedding=embeddings,
        persist_directory=persist_dir
    )
else:
    print("Loading existing vector database...")
    
    vector_store = Chroma(
        persist_directory=persist_dir,
        embedding_function=embeddings
    )

# query and retrieval 
query = "How to stay fit?"

results = vector_store.similarity_search(
    query,
    k=3   # number of chunks to retrieve
)

# create chat model
llm = ChatOllama(model="mistral:latest")

# combine context
context = "\n\n".join([doc.page_content for doc in results])

# messages (better than raw prompt)
messages = [
    (
        "system",
        "Answer the question using only the provided context. say I dont know if dont know the answer",
    ),
    ("human", f"Context:\n{context}\n\nQuestion: {query}"),
]

# invoke
response = llm.invoke(messages)

print("\nFinal Answer:\n")
print(response.content)
