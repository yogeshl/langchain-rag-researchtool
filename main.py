import os
import streamlit as st
import pickle
import time
from langchain import OpenAI
from langchain.chains import RetrievalQAWithSourcesChain
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.document_loaders import UnstructuredURLLoader
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import FAISS


from dotenv import load_dotenv
load_dotenv()


file_path = "my_faiss_index.index"
embeddings = OpenAIEmbeddings()


st.title("Research Tool 🕸")

st.sidebar.title("Article Urls")

urls = []
for i in range(3):
    url = st.sidebar.text_input(f"URL {i+1}")
    urls.append(url)

analyze_btn_clicked =  st.sidebar.button('Analyze Articles')

main_placeholder = st.empty()

llm = OpenAI(temperature=0.9, max_tokens=500)

if analyze_btn_clicked:
   #loading data
   main_placeholder.text("Loading Articles ⌛⌛⌛")
   loader = UnstructuredURLLoader(urls = urls)
   data = loader.load()

   # splitting data to chunks
   main_placeholder.text("Splitting content to chunks ⌛⌛⌛")
   text_splitter = RecursiveCharacterTextSplitter(
       separators=['\n\n','\n','.',','],
       chunk_size=1000
   )
   docs = text_splitter.split_documents(data)

   # Create embeddings and add to vector store
   main_placeholder.text("Embedding content and adding to vector index/store ⌛⌛⌛")
   #embeddings = OpenAIEmbeddings()
   vector_store = FAISS.from_documents(docs, embeddings)

   # Save the FAISS index to a  file
   vector_store.save_local("faiss_index")
   

query = main_placeholder.text_input("Question: ")
if query:
     
     if os.path.exists("faiss_index"):
        
        # with open(file_path, "rb") as f:
        #     vector_store = pickle.load(f)

        vector_store = FAISS.load_local("faiss_index", embeddings, allow_dangerous_deserialization=True)
        
        chain = RetrievalQAWithSourcesChain.from_llm(llm=llm, retriever=vector_store.as_retriever())
        result = chain({"question": query}, return_only_outputs = True)
        # result will be a dictionary of this format --> {"answer": "", "sources": [] }
        st.header("Answer")
        st.write(result["answer"])

        # Display sources, if available
        sources = result.get("sources", "")
        if sources:
            st.subheader("Sources :")
            sources_list = sources.split("\n")
            for source in sources_list:
                st.write(source)






