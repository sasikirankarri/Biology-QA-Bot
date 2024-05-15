
# Biology RAG Question and Answering System

This repository contains the implementation of a Biology Retrieval-Augmented Generation (RAG) Question and Answering system. This system utilizes state-of-the-art NLP models to answer questions based on biology textbook content. It leverages the Hugging Face transformers and FAISS for efficient similarity search in high-dimensional spaces.

## Features

- **Text Extraction from PDF**: Extracts text from specified pages of a biology textbook PDF.
- **FAISS Indexing**: Creates and uses FAISS indices to facilitate efficient retrieval from a large corpus.
- **RAG Question Answering**: Utilizes the RAG model from Hugging Face to generate answers based on the context provided by the retrieved documents.

## Prerequisites

Before you run the project, make sure you have the following installed:
- Python 3.8 or above
- Pip (Python package installer)

## Installation

Clone the repository and install the required Python packages:

```bash
git clone https://github.com/sasikirankarri/Biology-QA-Bot.git
cd Biology-QA-Bot
pip install -r requirements.txt
```

## Configuration

1. **API Key**: Ensure you have an API key from Hugging Face. Place this key in a JSON file within the `./env/` directory with the following format:

```json
{
  "api_key": "your_huggingface_api_key_here"
}
```

2. **Data Setup**: Place your input biology textbook PDF in the `./data/` directory or modify the `--input_data_path` argument when running the script.

## Usage

### Running the System

To run the system, use the following command:

```bash
python run_rag.py --query "Your question here" [other options]
```

### Options

You can customize the execution of the program by specifying the following options:

- `--input_data_path <path>`: Path to the directory containing the biology textbook PDF or extracted text. Default is `./data/`.
- `--start_page <number>`: Starting page number to extract text from the PDF. Default is `18`.
- `--end_page <number>`: Ending page number to extract text from the PDF. Default is `73`.
- `--index_path <path>`: Path where the FAISS index files are stored or will be created. Default is `./db_files/`.
- `--index_model <model_identifier>`: Model identifier for the sentence transformer model used in the FAISS index. Default is `sentence-transformers/all-MiniLM-l6-v2`.
- `--model_name <model_identifier>`: Model identifier on Hugging Face for the RAG model. Default is `mistralai/Mistral-7B-v0.1`.
- `--query <text>`: The question to be asked to the model. This is a required argument.

### Example

Here is an example of running the script:

```bash
python run_rag.py --query "What are the key functions of the cell membrane?" \
                          --input_data_path "./data/" \
                          --start_page 18 \
                          --end_page 73 \
                          --index_path "./db_files/" \
                          --index_model "sentence-transformers/all-MiniLM-l6-v2" \
                          --model_name "mistralai/Mistral-7B-v0.1"
```

## Contact

Sasi Kiran Karri - [sasikirankarri@outlook.com](mailto:sasikirankarri@outlook.com)

Project Link: [https://github.com/sasikirankarri/](https://github.com/sasikirankarri/Biology-QA-Bot)

## Acknowledgments

- [Hugging Face Transformers](https://huggingface.co/transformers/)
- [FAISS](https://github.com/facebookresearch/faiss)
- [Sentence Transformers](https://www.sbert.net/)
```
