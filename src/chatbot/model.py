from langchain.text_splitter import TokenTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores.faiss import FAISS
from transformers import AutoTokenizer, pipeline
from langchain import HuggingFacePipeline
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
import re

class RAG_Pipeline:
    """
    A pipeline to retrieve and generate answers based on a provided query and a vector store using the
    HuggingFace transformers library.

    This class initializes a language model for question answering and defines methods to generate and
    extract answers from relevant contexts.

    Attributes:
    data_path (str): The path to the dataset or where data files are stored.
    model_name (str): The name or path of the HuggingFace model to use for generating answers.
    tokenizer (AutoTokenizer): Tokenizer from the HuggingFace library based on the specified model.
    question_answerer (pipeline): A HuggingFace pipeline for text generation initialized with the model.
    llm (HuggingFacePipeline): A langchain HuggingFacePipeline instance for processing text.
    """

    def __init__(self, data_path, model_name):
        """
        Initializes the RAG_Pipeline with the specified dataset path and model name.

        Parameters:
        data_path (str): The path to the data files.
        model_name (str): The name or path of the model to use for text generation.
        """
        self.data_path = data_path
        self.model_name = model_name
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        #self.model = AutoModelForQuestionAnswering.from_pretrained(self.model_name)
        self.question_answerer = pipeline("text-generation", model=self.model_name, tokenizer=self.tokenizer, max_new_tokens=256)
        self.llm = HuggingFacePipeline(pipeline=self.question_answerer)

    def generate_answer(self, query, vector_store):
        """
        Generates an answer for the given query using the provided vector store to retrieve relevant contexts.

        Parameters:
        query (str): The question or query for which to generate an answer.
        vector_store (FAISS): The vector store used to retrieve relevant documents based on the query.

        Returns:
        str: The generated answer based on the query and the retrieved documents.
        """
        retriever = vector_store.as_retriever(search_kwargs={"k": 3})

        prompt_template = """Use the following pieces of context to answer the question at the end. Please follow the following rules:
                            1. If you don't know the answer, don't try to make up an answer. Just say "I can't find the final answer but you may want to check the following links".
                           
                              {context}

                              Question: {question}

                              Helpful Answer:
                              """

        PROMPT = PromptTemplate(template=prompt_template, input_variables=["context", "question"])
        qa = RetrievalQA.from_chain_type(llm=self.llm, chain_type="stuff", retriever=retriever, return_source_documents=False, chain_type_kwargs={"prompt": PROMPT})
        response = qa.invoke({"query": query})
        
        return response['result']

    def extract_answer(self, text):
        """
        Extracts a concise answer from the given text based on the predefined pattern 'Helpful Answer:'.

        Parameters:
        text (str): The text from which to extract the answer.

        Returns:
        str: The extracted answer, concatenated and cleaned from the provided text.
        """
        pattern = re.compile(r'Helpful Answer:\s*\n*(.*?)\n*(?=Helpful Answer:|$)', re.DOTALL)
        matches = pattern.findall(text)

        answer = ""
        for i, match in enumerate(matches, 1):
            answer += match.strip()

        return answer
