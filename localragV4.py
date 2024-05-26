import transformers
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import PyPDF2
import re
import tkinter as tk
from tkinter import filedialog

# Function to convert PDF to text
def convert_pdf_to_text():
  file_path = filedialog.askopenfilename(filetypes=[("PDF Files", "*.pdf")])
  if file_path:
    with open(file_path, 'rb') as pdf_file:
      pdf_reader = PyPDF2.PdfReader(pdf_file)
      text = ''
      for page in pdf_reader.pages:
        text += page.extract_text()
      # Normalize whitespace and clean up text
      text = re.sub(r'\s+', ' ', text).strip()
      return text
  return None

# Function to open a file and return its contents as a string
def open_file(filepath):
  with open(filepath, 'r', encoding='utf-8') as infile:
    return infile.read()

# Function to rewrite a query using a pre-trained summarization model
def rewrite_query(user_input, model_name="t5-base"):
  tokenizer = AutoTokenizer.from_pretrained(model_name)
  model = AutoModelForSeq2SeqLM.from_pretrained(model_name)

  # Prepare input for summarization
  inputs = tokenizer(user_input, return_tensors="pt")

  # Generate summary (rewritten query)
  summary_outputs = model.generate(**inputs)
  decoded_summary = tokenizer.batch_decode(summary_outputs, skip_special_tokens=True)[0]
  return decoded_summary

# Conversation loop
print("Starting conversation loop...")
conversation_history = []
system_message = "I am a helpful assistant that can summarize your documents and answer questions based on their content."

while True:
  user_input = input("Ask a question about your documents (or type 'quit' to exit): ")
  if user_input.lower() == 'quit':
    break

  # Load or convert document if needed
  document_text = None  # Replace with your document loading logic

  # Rewrite query using summarization model
  rewritten_query = rewrite_query(user_input)
  print(f"Original Query: {user_input}")
  print(f"Rewritten Query: {rewritten_query}")

  # Simulate answering the question based on rewritten query and document (replace with your logic)
  print("Answer: (Provide a relevant answer based on the document and rewritten query)")

  conversation_history.append({"role": "user", "content": user_input})
  conversation_history.append({"role": "assistant", "content": "Answer: (Your answer here)"})
