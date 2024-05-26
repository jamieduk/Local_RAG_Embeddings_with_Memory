import os
import tkinter as tk
from tkinter import filedialog
import PyPDF2
import re
import json
import pytesseract
from PIL import Image
import io
from pdf2image import convert_from_path


def extract_text_from_image(pdf_path):
    text=''
    images=convert_from_path(pdf_path)
    for img in images:
        text += pytesseract.image_to_string(img)
    return text

def upload_txtfile():
    file_path=filedialog.askopenfilename(filetypes=[("Text Files", "*.txt")])
    if file_path:
        with open(file_path, 'r', encoding="utf-8") as txt_file:
            text=txt_file.read()
            
            # Normalize whitespace and clean up text
            text=re.sub(r'\s+', ' ', text).strip()
            
            # Split text into chunks by sentences, respecting a maximum chunk size
            sentences=re.split(r'(?<=[.!?]) +', text)
            chunks=[]
            current_chunk=""
            for sentence in sentences:
                if len(current_chunk) + len(sentence) + 1 < 1000:
                    current_chunk += (sentence + " ").strip()
                else:
                    chunks.append(current_chunk)
                    current_chunk=sentence + " "
            if current_chunk:
                chunks.append(current_chunk)
                
            with open("vault.txt", "a", encoding="utf-8") as vault_file:
                for chunk in chunks:
                    vault_file.write(chunk.strip() + "\n")
            print(f"Text file content appended to vault.txt with each chunk on a separate line.")


def convert_pdf_to_text():
    file_path=filedialog.askopenfilename(filetypes=[("PDF Files", "*.pdf")])
    if file_path:
        text=extract_text_from_image(file_path)
        
        # Normalize whitespace and clean up text
        text=re.sub(r'\s+', ' ', text).strip()
        
        # Split text into chunks by sentences, respecting a maximum chunk size
        sentences=re.split(r'(?<=[.!?]) +', text)
        chunks=[]
        current_chunk=""
        for sentence in sentences:
            if len(current_chunk) + len(sentence) + 1 < 1000:
                current_chunk += (sentence + " ").strip()
            else:
                chunks.append(current_chunk)
                current_chunk=sentence + " "
        if current_chunk:
            chunks.append(current_chunk)
        
        with open("vault.txt", "a", encoding="utf-8") as vault_file:
            for chunk in chunks:
                vault_file.write(chunk.strip() + "\n")
        print(f"PDF content appended to vault.txt with each chunk on a separate line.")


# Function to upload a JSON file and append to vault.txt
def upload_jsonfile():
    file_path=filedialog.askopenfilename(filetypes=[("JSON Files", "*.json")])
    if file_path:
        with open(file_path, 'r', encoding="utf-8") as json_file:
            data=json.load(json_file)
            
            text=json.dumps(data, ensure_ascii=False)
            
            text=re.sub(r'\s+', ' ', text).strip()
            
            sentences=re.split(r'(?<=[.!?]) +', text)
            chunks=[]
            current_chunk=""
            for sentence in sentences:
                if len(current_chunk) + len(sentence) + 1 < 1000:
                    current_chunk += (sentence + " ").strip()
                else:
                    chunks.append(current_chunk)
                    current_chunk=sentence + " "
            if current_chunk:
                chunks.append(current_chunk)
                
            with open("vault.txt", "a", encoding="utf-8") as vault_file:
                for chunk in chunks:
                    vault_file.write(chunk.strip() + "\n")
            print(f"JSON file content appended to vault.txt with each chunk on a separate line.")

# Create the main window
root=tk.Tk()
root.title("Upload .pdf, .txt, or .json")

pdf_button=tk.Button(root, text="Upload PDF", command=convert_pdf_to_text)
pdf_button.pack(pady=10)

txt_button=tk.Button(root, text="Upload Text File", command=upload_txtfile)
txt_button.pack(pady=10)

json_button=tk.Button(root, text="Upload JSON File", command=upload_jsonfile)
json_button.pack(pady=10)

root.mainloop()

