# from docx import Document
# import re

# def preprocess_text(text):
#     # Normalize spaces and remove extra blank lines
#     text = text.strip()
#     text = re.sub(r'\s+', ' ', text)  # Replace multiple spaces/newlines with single space
#     return text

# def extract_docx_to_txt(input_file, output_file):
#     doc = Document(input_file)
#     with open(output_file, 'w', encoding='utf-8') as f:
#         for para in doc.paragraphs:
#             if para.text.strip():
#                 # If the paragraph has bold text (title)
#                 if any(run.bold for run in para.runs):
#                     f.write("\n=== TITLE: " + para.text.strip() + " ===\n")
#                 else:
#                     f.write(preprocess_text(para.text) + "\n")

#         # Handle tables
#         for table_index, table in enumerate(doc.tables):
#             f.write(f"\n=== TABLE {table_index + 1} ===\n")
#             for row in table.rows:
#                 row_text = '\t'.join(cell.text.strip() for cell in row.cells)
#                 f.write(preprocess_text(row_text) + "\n")

#     print(f"✔️ Successfully saved to {output_file}")

# # Usage
# input_docx = "DSEU_Admission Brochure 2025 final(1).docx"
# output_txt = "output.txt"
# extract_docx_to_txt(input_docx, output_txt)






from langchain.schema import Document
import re

def load_structured_txt(file_path):
    documents = []
    current_title = ""
    current_type = "text"
    buffer = []

    with open(file_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()

            if not line:
                continue

            # Title detection
            title_match = re.match(r"=== TITLE: (.+?) ===", line)
            if title_match:
                if buffer:
                    documents.append(Document(page_content="\n".join(buffer), metadata={"title": current_title, "type": current_type}))
                    buffer = []
                current_title = title_match.group(1)
                current_type = "text"
                continue

            # Table detection
            table_match = re.match(r"=== TABLE (\d+) ===", line)
            if table_match:
                if buffer:
                    documents.append(Document(page_content="\n".join(buffer), metadata={"title": current_title, "type": current_type}))
                    buffer = []
                current_title = f"Table {table_match.group(1)}"
                current_type = "table"
                continue

            buffer.append(line)

        if buffer:
            documents.append(Document(page_content="\n".join(buffer), metadata={"title": current_title, "type": current_type}))

    return documents
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings

docs = load_structured_txt("output.txt")

embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
vectorstore = FAISS.from_documents(docs, embeddings)
vectorstore.save_local("data/defaultllm_docs")
