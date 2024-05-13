import fitz  # PyMuPDF
import re
import os

class PDFExtractor:
    """
    A class to extract text from a PDF file, clean the text, and handle file operations related to the text.

    Attributes:
    data_path (str): The base directory path where PDF files and text files are stored.
    """
    
    def __init__(self, data_path):
        """
        Initializes the PDFExtractor with a data path and page range.

        Parameters:
        data_path (str): The base directory path where the PDF and resulting text files are stored.
        """
        self.data_path = data_path


    def extract_text_from_pdf(self, pdf_file, start_page_number, end_page_number):
        """
        Extracts text from a specified range of pages in a PDF file.

        Parameters:
        pdf_file (str): The file name of the PDF document within the data_path directory.
        start_page_number (int): The starting page number for extraction.
        end_page_number (int): The ending page number for extraction.

        Returns:
        str: The concatenated text extracted from the specified page range.
        """
        pdf_path = os.path.join(self.data_path, pdf_file)
        document = fitz.open(pdf_path)
        text = ""
        for page in document[start_page_number:end_page_number]:
            text += page.get_text()
        return text

    @staticmethod
    def clean_text(text):
        """
        Cleans and segments the input text into paragraphs by removing excessive whitespace and splitting on double newlines.

        Parameters:
        text (str): The input text to be cleaned and segmented.

        Returns:
        list of str: A list of paragraphs from the input text.
        """
        text = re.sub(r'\s+', ' ', text).strip()
        segments = text.split('\n\n')
        extracted_text = [segment.strip() for segment in segments if segment.strip()]
        return str(extracted_text[0])

    def save_text_local(self, file_name, text):
        """
        Saves the provided text into a file within the data_path directory.

        Parameters:
        file_name (str): The name of the file where the text will be saved.
        text (str): The text to be saved.
        """
        cleaned_text = self.clean_text(text)
        file_path = os.path.join(self.data_path, file_name)
        with open(file_path, 'w') as f:
            f.write(cleaned_text)

    def load_text(self, file_name):
        """
        Loads and returns the text from a specified file within the data_path directory.

        Parameters:
        file_name (str): The name of the file from which to load the text.

        Returns:
        str: The text loaded from the specified file.
        """
        file_path = os.path.join(self.data_path, file_name)
        with open(file_path, "r") as f:
            extracted_text = f.read()
        return extracted_text
