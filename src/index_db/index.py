from langchain.embeddings import HuggingFaceEmbeddings
from langchain.text_splitter import TokenTextSplitter
from langchain_community.vectorstores import FAISS

class Faiss_Index:
    """
    A class to create, manage, and utilize a FAISS index for efficient similarity search in high-dimensional spaces.

    This class uses embeddings from a HuggingFace model to transform text into vectors and store these vectors in a
    FAISS index for quick similarity searches.

    Attributes:
    ind_path (str): The path where the FAISS index is stored or will be saved.
    model (str): The name or path of the pre-trained HuggingFace model used for embedding the text.
    model_kwargs (dict): Additional keyword arguments for configuring the model (e.g., setting device to 'cpu').
    encode_kwargs (dict): Additional keyword arguments for the encoding process (e.g., normalizing embeddings).
    embeddings (HuggingFaceEmbeddings): An instance of HuggingFaceEmbeddings to convert text into embeddings.
    """

    def __init__(self, ind_path, model):
        """
        Initializes the Faiss_Index with a path to store the index and a model for embeddings.

        Parameters:
        ind_path (str): The filesystem path where the FAISS index is or will be saved.
        model (str): The name or path of the HuggingFace model to use for generating embeddings.
        """
        self.ind_path = ind_path
        self.model = model
        self.model_kwargs = {'device': 'cpu'}
        self.encode_kwargs = {'normalize_embeddings': False}
        self.embeddings = HuggingFaceEmbeddings(
            model_name=self.model,
            model_kwargs=self.model_kwargs,
            encode_kwargs=self.encode_kwargs
        )

    def create_index_text(self, extracted_text_path):
        """
        Creates a FAISS index from the provided text by embedding it using the specified model.

        Parameters:
        extracted_text (str): The input text to embed and index.

        Returns:
        FAISS: A FAISS vector store containing the embeddings of the split input text.
        """
        with open(extracted_text_path,"r") as f:
            extracted_text = f.read()
            
        #print(len(extracted_text))
        
        text_splitter = TokenTextSplitter(chunk_size=100, chunk_overlap=0)
        text_splited = text_splitter.split_text(extracted_text)
        #print(text_splited)
        vectorstore = FAISS.from_texts(text_splited, self.embeddings)

        return vectorstore

    def index_to_local(self, vectorstore):
        """
        Saves the provided FAISS vector store to a file at the specified index path.

        Parameters:
        vectorstore (FAISS): The FAISS vector store to save to disk.
        """
        FAISS.save_local(vectorstore, self.ind_path)

    def local_to_index(self):
        """
        Loads a FAISS index from the local file system into a FAISS vector store.

        Returns:
        FAISS: The loaded FAISS vector store.
        """
        vectorstore = FAISS.load_local(self.ind_path, self.embeddings, allow_dangerous_deserialization=True)
        return vectorstore
