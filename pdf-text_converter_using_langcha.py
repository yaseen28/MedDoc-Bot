"""This file extracts the texts from PDF document using langchain library. !! In addition you can use PDF24 application to convert PDF to text.

The extracted files are then copied to word document for further processing such as Aranging in order and cleaning the text

Install following libraries before running this code:

pip install langchain

pip install unstructured_inference

pip install pikepdf

pip install pypdf

pip install unstructured
"""


from langchain.document_loaders import UnstructuredFileLoader

def extract_text_with_langchain_pdf(pdf_file):

    loader = UnstructuredFileLoader(pdf_file)
    documents = loader.load()
    pdf_pages_content = '\n'.join(doc.page_content for doc in documents)

    return pdf_pages_content

text_with_langchain_files = extract_text_with_langchain_pdf("guidelines.pdf") # Replace with your file path
print(text_with_langchain_files)
