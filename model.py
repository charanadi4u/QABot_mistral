import logging
from langchain.document_loaders import PyPDFLoader, DirectoryLoader
from langchain.prompts import PromptTemplate
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS
from langchain.chains import RetrievalQA
from gpt4all import GPT4All
from langchain.llms import GPT4All

class QABot:
    DB_FAISS_PATH = 'vectorstore/db_faiss'

    def __init__(self):
        # Configure logging
        self.logger = logging.getLogger('QABot')
        logging.basicConfig(level=logging.INFO)

        # Load components
        self.logger.info("Initializing the QA Bot")
        self.embeddings = self.load_embeddings()
        self.db = self.load_db()
        self.llm = self.load_llm()
        self.qa_prompt = self.set_custom_prompt()
        self.qa = self.retrieval_qa_chain(self.llm, self.qa_prompt, self.db)
    
    def set_custom_prompt(self):
        """
        Prompt template for QA retrieval for each vectorstore
        """
        custom_prompt_template =  """Use the following pieces of information to answer the user's question.
                                     If you don't know the answer, just say that you don't know, don't try to make up an answer.

                                     Context: {context}
                                     Question: {question}

                                  Only return the helpful answer below and nothing else.
                                  Helpful answer:
                                  """
        prompt = PromptTemplate(template=custom_prompt_template,
                                input_variables=['context', 'question'])
        return prompt

    def retrieval_qa_chain(self, llm, prompt, db):
        qa_chain = RetrievalQA.from_chain_type(llm=llm,
                                               chain_type='stuff',
                                               retriever=db.as_retriever(search_kwargs={'k': 2}),
                                               return_source_documents=True,
                                               chain_type_kwargs={'prompt': prompt}
                                               )
        return qa_chain

    def load_embeddings(self):
        self.logger.info("Loading embeddings")
        embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2",
                                           model_kwargs={'device': 'cpu'})
        return embeddings

    def load_db(self):
        self.logger.info("Loading FAISS database")
        db = FAISS.load_local(self.DB_FAISS_PATH, self.embeddings)
        return db

    def load_llm(self):
        self.logger.info("Loading language model")
        llm = GPT4All(model="F:\GPT4all\mistral-7b-instruct-v0.1.Q4_0.gguf",
                      allow_download=False,
                      device='cpu')
        return llm

    def final_result(self, query):
        self.logger.info(f"Processing query: {query}")
        response = self.qa({'query': query})
        return response

# Example of using the class
# if __name__ == "__main__":
#     bot = QABot()
#     query = "What is the capital of France?"
#     result = bot.final_result(query)
#     print(result)
