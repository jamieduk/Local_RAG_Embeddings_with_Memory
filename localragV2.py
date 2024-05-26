import torch
import ollama
import os
import json
from openai import OpenAI
import argparse
from datetime import datetime
import io
import PyPDF2
import pytesseract
from PIL import Image
import re
import tkinter as tk
from tkinter import filedialog

# ANSI escape codes for colors
PINK='\033[95m'
CYAN='\033[96m'
YELLOW='\033[93m'
NEON_GREEN='\033[92m'
RESET_COLOR='\033[0m'

# Function to extract text from images using OCR
def extract_text_from_image(page):
    text=''
    for image_obj in page.images:
        xref=image_obj[0]
        base_image=page.get_xobject(xref)
        image=Image.open(io.BytesIO(base_image.stream.get_data()))
        text += pytesseract.image_to_string(image)
    return text

# Function to convert PDF to text and append to vault.txt
def convert_pdf_to_text():
    file_path=filedialog.askopenfilename(filetypes=[("PDF Files", "*.pdf")])
    if file_path:
        with open(file_path, 'rb') as pdf_file:
            pdf_reader=PyPDF2.PdfReader(pdf_file)
            text=''
            for page in pdf_reader.pages:
                page_text=page.extract_text()
                if not page_text:
                    page_text=extract_text_from_image(page)
                if page_text:
                    text += page_text + " "
            
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

# Function to open a file and return its contents as a string
def open_file(filepath):
    with open(filepath, 'r', encoding='utf-8') as infile:
        return infile.read()

# Function to get relevant context from the vault based on user input
def get_relevant_context(rewritten_input, vault_embeddings, vault_content, top_k=3):
    if vault_embeddings.nelement() == 0:  # Check if the tensor has any elements
        return []
    # Encode the rewritten input
    input_embedding=ollama.embeddings(model='mxbai-embed-large', prompt=rewritten_input)["embedding"]
    # Compute cosine similarity between the input and vault embeddings
    cos_scores=torch.cosine_similarity(torch.tensor(input_embedding).unsqueeze(0), vault_embeddings)
    # Adjust top_k if it's greater than the number of available scores
    top_k=min(top_k, len(cos_scores))
    # Sort the scores and get the top-k indices
    top_indices=torch.topk(cos_scores, k=top_k)[1].tolist()
    # Get the corresponding context from the vault
    relevant_context=[vault_content[idx].strip() for idx in top_indices]
    return relevant_context

def rewrite_query(user_input_json, conversation_history, ollama_model):
    user_input=json.loads(user_input_json)["Query"]
    context="\n".join([f"{msg['role']}: {msg['content']}" for msg in conversation_history[-2:]])
    prompt=f"""Rewrite the following query by incorporating relevant context from the conversation history.
    The rewritten query should:
    
    - Preserve the core intent and meaning of the original query
    - Expand and clarify the query to make it more specific and informative for retrieving relevant context
    - Avoid introducing new topics or queries that deviate from the original query
    - DONT EVER ANSWER the Original query, but instead focus on rephrasing and expanding it into a new query
    
    Return ONLY the rewritten query text, without any additional formatting or explanations.
    
    Conversation History:
    {context}
    
    Original query: [{user_input}]
    
    Rewritten query: 
    """
    response=client.chat.completions.create(
        model=ollama_model,
        messages=[{"role": "system", "content": prompt}],
        max_tokens=200,
        n=1,
        temperature=0.1,
    )
    rewritten_query=response.choices[0].message.content.strip()
    return json.dumps({"Rewritten Query": rewritten_query})



def ollama_chat(user_input, system_message, vault_embeddings, vault_content, ollama_model, conversation_history):
    conversation_history.append({"role": "user", "content": user_input})
    
    if len(conversation_history) > 1:
        query_json={
            "Query": user_input,
            "Rewritten Query": ""
        }
        rewritten_query_json=rewrite_query(json.dumps(query_json), conversation_history, ollama_model)
        rewritten_query_data=json.loads(rewritten_query_json)
        rewritten_query=rewritten_query_data["Rewritten Query"]
        print(PINK + "Original Query: " + user_input + RESET_COLOR)
        print(PINK + "Rewritten Query: " + rewritten_query + RESET_COLOR)
    else:
        rewritten_query=user_input
    
    relevant_context=get_relevant_context(rewritten_query, vault_embeddings, vault_content)
    if relevant_context:
        context_str="\n".join(relevant_context)
        print("Context Pulled from Documents: \n\n" + CYAN + context_str + RESET_COLOR)
    else:
        print(CYAN + "No relevant context found." + RESET_COLOR)
    
    user_input_with_context=user_input
    if relevant_context:
        user_input_with_context=user_input + "\n\nRelevant Context:\n" + context_str
    
    conversation_history[-1]["content"]=user_input_with_context
    
    messages=[
        {"role": "system", "content": system_message},
        *conversation_history
    ]
    
    response=client.chat.completions.create(
        model=ollama_model,
        messages=messages,
        max_tokens=2000,
    )
    
    conversation_history.append({"role": "assistant", "content": response.choices[0].message.content})
    
    return response.choices[0].message.content

# Parse command-line arguments
print(NEON_GREEN + "Parsing command-line arguments..." + RESET_COLOR)
parser=argparse.ArgumentParser(description="Ollama Chat")
parser.add_argument("--model", default="dolphin-llama3:latest", help="Ollama model to use (default: llama3)") # llama3
args=parser.parse_args()

# Configuration for the Ollama API client
print(NEON_GREEN + "Initializing Ollama API client..." + RESET_COLOR)
client=OpenAI(
    base_url='http://localhost:11434/v1',
   # model="dolphin-llama3:latest",  # llama3:latest  mistral
    api_key='NA'
)

# Load the vault content
print(NEON_GREEN + "Loading vault content..." + RESET_COLOR)
vault_content=[]
vault_file_path="vault.txt"
if os.path.exists(vault_file_path):
    vault_modified_time=os.path.getmtime(vault_file_path)
    with open(vault_file_path, "r", encoding='utf-8') as vault_file:
        vault_content=vault_file.readlines()
    # Check if embeddings file exists and is up to date, otherwise regenerate it
    embeddings_file_path="vault_embeddings.pt"
    regenerate_embeddings=True
    if os.path.exists(embeddings_file_path):
        embeddings_modified_time=os.path.getmtime(embeddings_file_path)
        if embeddings_modified_time > vault_modified_time:
            regenerate_embeddings=False
    if regenerate_embeddings:
        print(NEON_GREEN + "Generating embeddings for the vault content..." + RESET_COLOR)
        vault_embeddings=[]
        for content in vault_content:
            response=ollama.embeddings(model='mxbai-embed-large', prompt=content)
            vault_embeddings.append(response["embedding"])
        # Save embeddings to file
        torch.save(torch.tensor(vault_embeddings), embeddings_file_path)
else:
    print(YELLOW + "Vault file not found." + RESET_COLOR)

# Load or regenerate embeddings
if os.path.exists(embeddings_file_path):
    print(NEON_GREEN + "Loading embeddings from file..." + RESET_COLOR)
    vault_embeddings_tensor=torch.load(embeddings_file_path)
else:
    print(YELLOW + "Embeddings file not found or outdated. Regenerating embeddings..." + RESET_COLOR)
    # Regenerate embeddings
    vault_embeddings_tensor=torch.tensor(vault_embeddings)

# Conversation loop
print("Starting conversation loop...")
conversation_history=[]
system_message="You are a helpful assistant that is an expert at extracting the most useful information from a given text. Also bring in extra relevant infromation to the user query from outside the given context."

while True:
    user_input=input(YELLOW + "Ask a query about your documents (or type 'quit' to exit): " + RESET_COLOR)
    if user_input.lower() == 'quit':
        break
    
    response=ollama_chat(user_input, system_message, vault_embeddings_tensor, vault_content, args.model, conversation_history)
    print(NEON_GREEN + "Response: \n\n" + response + RESET_COLOR)



