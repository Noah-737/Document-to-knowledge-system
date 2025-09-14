import os
from bs4 import BeautifulSoup
import PyPDF2

def extract_text_from_file(file_path: str) -> str:
    """
    Extracts text from a file based on its extension.

    Args:
        file_path: The path to the file.

    Returns:
        The extracted text as a string.
    """
    _, extension = os.path.splitext(file_path)
    text = ""

    if extension.lower() == ".pdf":
        try:
            with open(file_path, "rb") as f:
                reader = PyPDF2.PdfReader(f)
                for page in reader.pages:
                    text += page.extract_text() or ""
        except Exception as e:
            print(f"Error reading PDF {file_path}: {e}")
            return ""
    elif extension.lower() in [".html", ".htm"]:
        try:
            with open(file_path, "r", encoding="utf-8") as f:
                soup = BeautifulSoup(f, "html.parser")
                text = soup.get_text()
        except Exception as e:
            print(f"Error reading HTML {file_path}: {e}")
            return ""
    elif extension.lower() == ".txt":
        try:
            with open(file_path, "r", encoding="utf-8") as f:
                text = f.read()
        except Exception as e:
            print(f"Error reading TXT {file_path}: {e}")
            return ""
    else:
        print(f"Unsupported file type: {extension}")
        return ""

    return text


from langchain.text_splitter import RecursiveCharacterTextSplitter

def chunk_text(text: str) -> list[str]:
    """
    Splits the text into chunks of a specified size.

    Args:
        text: The text to be chunked.

    Returns:
        A list of text chunks.
    """
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
        length_function=len
    )
    chunks = text_splitter.split_text(text)
    return chunks
