import os
import re
import json
import torch
import httpx
import PyPDF2
import io
from PIL import Image
import pytesseract
from tkinter import filedialog
from spellchecker import SpellChecker
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import argparse
import time
import subprocess
import platform

# Global variable to store the cache
CACHE_FILE="cache.json"
query_cache={}



def speak_response(file_path):
    if os.path.exists(file_path) and os.path.getsize(file_path) > 0:
        with open(file_path, 'r') as file:
            text=file.read()
            if platform.system() == "Windows":
                subprocess.call(['espeak', text], shell=True)
            else:
                subprocess.call(['espeak', text])



def load_cache():
    global query_cache
    if os.path.exists(CACHE_FILE):
        with open(CACHE_FILE, "r") as file:
            query_cache=json.load(file)

def save_cache():
    global query_cache
    with open(CACHE_FILE, "w") as file:
        json.dump(query_cache, file)

# Load cache at the beginning of the program
load_cache()


# Placeholder for Ollama API client functions
class Ollama:
    @staticmethod
    def embeddings(model, prompt):
        # Replace with the actual API call to get embeddings
        return {"embedding": [0.0] * 768}  # Example embedding

ollama=Ollama()

# Function to extract text from images using OCR
def extract_text_from_image(page):
    text=''
    for image_obj in page.images:
        xref=image_obj[0]
        base_image=page.get_xobject(xref)
        image=Image.open(io.BytesIO(base_image.stream.get_data()))
        text += pytesseract.image_to_string(image)
    return text

# Function to convert PDF to text and replace vault.txt
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

            # Initialize the spell checker
            spell=SpellChecker()

            # Clean and spell check each chunk
            cleaned_chunks=[]
            for chunk in chunks:
                words=chunk.split()
                corrected_words=[spell.correction(word) for word in words]
                corrected_text=' '.join(corrected_words)
                cleaned_text=re.sub(r'[^a-zA-Z0-9\s,.!?]', '', corrected_text)
                cleaned_chunks.append(cleaned_text)

            with open("vault.txt", "w", encoding="utf-8") as vault_file:
                for chunk in cleaned_chunks:
                    vault_file.write(chunk.strip() + "\n")
            print(f"PDF content replaced in vault.txt with each chunk on a separate line.")

# Function to open a file and return its contents as a string
def open_file(filepath):
    with open(filepath, 'r', encoding='utf-8') as infile:
        return infile.read()




def get_relevant_context(rewritten_input, vault_embeddings, vault_content, top_k=3):
    print(f"vault_embeddings.shape: {vault_embeddings.shape}")
    print(f"vault_content length: {len(vault_content)}")
    
    if vault_embeddings.nelement() == 0:
        return []

    input_embedding=ollama.embeddings(model='mxbai-embed-large', prompt=rewritten_input)["embedding"]
    input_embedding_tensor=torch.tensor(input_embedding).unsqueeze(0)  # Add batch dimension

    # Ensure input embedding tensor has the same dimensionality as vault embeddings
    if input_embedding_tensor.shape[1] != vault_embeddings.shape[1]:
        target_dim=vault_embeddings.shape[1]
        input_embedding_tensor=input_embedding_tensor[:, :target_dim]  # Trim to match the dimensions
        if input_embedding_tensor.shape[1] < target_dim:
            padding=target_dim - input_embedding_tensor.shape[1]
            input_embedding_tensor=torch.nn.functional.pad(input_embedding_tensor, (0, padding))  # Pad to match the dimensions

    cos_scores=torch.nn.functional.cosine_similarity(input_embedding_tensor, vault_embeddings, dim=-1)
    top_k=min(top_k, len(cos_scores))
    top_indices=torch.topk(cos_scores, k=top_k)[1].tolist()

    print(f"top_indices: {top_indices}")
    
    # Ensure indices are within valid range
    relevant_context=[vault_content[idx].strip() for idx in top_indices if idx < len(vault_content)]
    
    return relevant_context





def rewrite_query(user_input_json, conversation_history, ollama_model, client):
    user_input=json.loads(user_input_json)["Query"]
    context="\n".join([f"{msg['role']}: {msg['content']}" for msg in conversation_history[-2:]])
    prompt=f"""Rewrite the following query by incorporating relevant context from the conversation history.
    The rewritten query should:
    - Ensure all text from llm aka ollama or any other models output should auto correct miss-spelt words (auto correct input and output)
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

    response=client.post(
        f"http://localhost:11434/v1/completions",
        json={
            "model": ollama_model,
            "prompt": prompt,
            "max_tokens": 200, # 200
        },
        timeout=600.0
    )

    if response.status_code == 200:
        rewritten_query=response.json()["choices"][0]["text"].strip()
        with open("rewritten_queries.txt", "w", encoding="utf-8") as rewritten_file:
            rewritten_file.write(rewritten_query + "\n")
        print("Rewritten query saved to rewritten_queries.txt")
        return json.dumps({"Rewritten Query": rewritten_query})
    else:
        print(f"HTTP request failed with status code {response.status_code}.")
        print(f"Response content: {response.content.decode('utf-8')}")
        return None



def ollama_chat(user_input, system_message, vault_embeddings, vault_content, ollama_model, conversation_history, client):
    global query_cache

    conversation_history.append({"role": "user", "content": user_input})

    if len(conversation_history) > 1:
        query_json={
            "Query": user_input,
            "Rewritten Query": ""
        }
        rewritten_query_json=rewrite_query(json.dumps(query_json), conversation_history, ollama_model, client)
        if rewritten_query_json:
            rewritten_query_data=json.loads(rewritten_query_json)
            rewritten_query=rewritten_query_data.get("Rewritten Query", "")  # Use get() to handle missing key
            print(f"Original Query: {user_input}")
            print(f"Rewritten Query: {rewritten_query}")
        else:
            print("Failed to rewrite the query.")
            rewritten_query=user_input  # Use original query if rewriting fails
    else:
        rewritten_query=user_input

    # Check if the user input is in the cache
    if user_input in query_cache:
        print("Using cached response.")
        assistant_response=query_cache[user_input]
        conversation_history.append({"role": "assistant", "content": assistant_response})
        return assistant_response

    # If not found in cache, proceed with the normal flow
    relevant_context=get_relevant_context(rewritten_query, vault_embeddings, vault_content)
    if relevant_context:
        context_str="\n".join(relevant_context)
        print("Context Pulled from Documents: \n\n" + context_str)
    else:
        print("No relevant context found.")

    user_input_with_context=user_input
    if relevant_context:
        user_input_with_context=user_input + "\n\nRelevant Context:\n" + context_str

    # Write user input with context to a file
    with open("user_input_with_context.txt", "w", encoding="utf-8") as input_file:
        input_file.write(user_input_with_context + "\n")

    conversation_history[-1]["content"]=user_input_with_context

    messages=[
        {"role": "system", "content": system_message},
        *conversation_history
    ]

    response=client.post(
        f"http://localhost:11434/v1/chat/completions",
        json={
            "model": ollama_model,
            "messages": messages,
            "max_tokens": 2000,
        },
        timeout=660.0
    )

    if response.status_code == 200:
        content=response.json()
        assistant_response=content["choices"][0]["message"]["content"]
        conversation_history.append({"role": "assistant", "content": assistant_response})

        # Write assistant response to a file
        with open("assistant_response.txt", "w", encoding="utf-8") as response_file:
            response_file.write(assistant_response + "\n")

        # Update the cache with the response
        query_cache[user_input]=assistant_response

        return assistant_response
    else:
        print(f"HTTP request failed with status code {response.status_code}.")
        return None




MAX_RETRIES=5

def ollama_chat_with_retry(user_input, system_message, vault_embeddings, vault_content, ollama_model, conversation_history, client):
    retries=0
    while retries < MAX_RETRIES:
        try:
            response=ollama_chat(user_input, system_message, vault_embeddings, vault_content, ollama_model, conversation_history, client)
            return response
        except httpx.ReadTimeout:
            retries += 1
            print(f"ReadTimeout occurred. Retrying... ({retries}/{MAX_RETRIES})")
            time.sleep(1)  # Wait for 1 second before retrying
    print("Max retries reached. Exiting...")
    return None





def main():
    start_time=time.time()  # Start the timer
    
    parser=argparse.ArgumentParser(description="Ollama Chat")
    parser.add_argument("--model", default="dolphin-llama3:latest", help="Ollama model to use (default: llama3)")
    args=parser.parse_args()

    client=httpx.Client(base_url='http://localhost:11434/v1', timeout=600.0)  # Increase timeout to 10 minutes

    vault_content=[]
    vault_file_path="vault.txt"
    vault_modified_time=0  # Initialize vault_modified_time here
    if os.path.exists(vault_file_path):
        vault_modified_time=os.path.getmtime(vault_file_path)
        with open(vault_file_path, "r", encoding="utf-8") as vault_file:
            vault_content=vault_file.readlines()


    embeddings_file_path="vault_embeddings.pt"
    regenerate_embeddings=True
    if os.path.exists(embeddings_file_path):
        embeddings_modified_time=os.path.getmtime(embeddings_file_path)
        if embeddings_modified_time > vault_modified_time:
            regenerate_embeddings=False
    if regenerate_embeddings:
        print("Generating embeddings for the vault content...")
        vault_embeddings=[]
        for content in vault_content:
            response=ollama.embeddings(model='mxbai-embed-large', prompt=content)
            vault_embeddings.append(response["embedding"])
        vault_embeddings_tensor=torch.tensor(vault_embeddings).cuda() if torch.cuda.is_available() else torch.tensor(vault_embeddings)
        torch.save(vault_embeddings_tensor, embeddings_file_path)
    else:
        vault_embeddings_tensor=torch.load(embeddings_file_path)
        if torch.cuda.is_available():
            vault_embeddings_tensor=vault_embeddings_tensor.cuda()

    system_message="You are an AI assistant here to help with queries."
    conversation_history=[]

    print("Starting conversation loop...")
    while True:
        user_input=input("Ask a query about your documents (or type 'quit' to exit): ")
        if user_input.lower() == 'quit':
            break
        response=ollama_chat(user_input, system_message, vault_embeddings_tensor, vault_content, args.model, conversation_history, client)
        if response:
            end_time=time.time()  # End the timer
            elapsed_time=end_time - start_time  # Calculate the elapsed time
            minutes=int(elapsed_time // 60)
            seconds=int(elapsed_time % 60)
            print(f"\nTotal elapsed time: {minutes} minutes and {seconds} seconds.")
            print("Assistant response:\n" + response)
			response_file='assistant_response.txt'
			
			# Check if the response file exists and is not empty
			if os.path.exists(response_file) and os.path.getsize(response_file) > 0:
				print(f"Reading from {response_file}...")
				speak_response(response_file)
			else:
				print(f"{response_file} does not exist or is empty.")
    client.close()  # Close the HTTP client at the end

if __name__ == "__main__":
    main()
