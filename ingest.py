from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS
from langchain.document_loaders import PyPDFLoader, DirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter 

class VectorDatabaseCreator:
    def __init__(self, data_path='data/', db_faiss_path='vectorstore/db_faiss'):
        self.data_path = data_path
        self.db_faiss_path = db_faiss_path

    def create_vector_db(self):
        loader = DirectoryLoader(self.data_path,
                                 glob='*.pdf',
                                 loader_cls=PyPDFLoader)

        documents = loader.load()
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=500,
                                                       chunk_overlap=50)
        texts = text_splitter.split_documents(documents)

        embeddings = HuggingFaceEmbeddings(model_name='sentence-transformers/all-MiniLM-L6-v2',
                                           model_kwargs={'device': 'cpu'})

        db = FAISS.from_documents(texts, embeddings)
        db.save_local(self.db_faiss_path)

# Usage
if __name__ == "__main__":
    vector_db_creator = VectorDatabaseCreator()
    vector_db_creator.create_vector_db()