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
from spellchecker import SpellChecker

def extract_text_from_image(pdf_path):
    text=''
    images=convert_from_path(pdf_path)
    for img in images:
        text += pytesseract.image_to_string(img)
    return text

def spell_check_and_correct(text):
    spell=SpellChecker()
    words=text.split()
    corrected_words=[spell.correction(word) or word for word in words]
    corrected_text=' '.join(corrected_words)
    return corrected_text

def process_and_append_text(text):
    # Spell check and correct the text
    corrected_text=spell_check_and_correct(text)

    # Normalize whitespace and clean up text
    corrected_text=re.sub(r'\s+', ' ', corrected_text).strip()

    # Split text into chunks by sentences, respecting a maximum chunk size
    sentences=re.split(r'(?<=[.!?]) +', corrected_text)
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
    print(f"Content saved to vault.txt with each chunk on a separate line.")

def upload_txtfile():
    file_path=filedialog.askopenfilename(filetypes=[("Text Files", "*.txt")])
    if file_path:
        with open(file_path, 'r', encoding="utf-8") as txt_file:
            text=txt_file.read()
            process_and_append_text(text)

def convert_pdf_to_text():
    file_path=filedialog.askopenfilename(filetypes=[("PDF Files", "*.pdf")])
    if file_path:
        text=extract_text_from_image(file_path)
        process_and_append_text(text)

def upload_jsonfile():
    file_path=filedialog.askopenfilename(filetypes=[("JSON Files", "*.json")])
    if file_path:
        with open(file_path, 'r', encoding="utf-8") as json_file:
            data=json.load(json_file)
            text=json.dumps(data, ensure_ascii=False)
            process_and_append_text(text)

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

