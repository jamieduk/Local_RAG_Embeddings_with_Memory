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
    if vault_embeddings.nelement() == 0:
        return []

    input_embedding=ollama.embeddings(model='mxbai-embed-large', prompt=rewritten_input)["embedding"]
    input_embedding_tensor=torch.tensor(input_embedding).cuda() if torch.cuda.is_available() else torch.tensor(input_embedding)
    input_embedding_tensor=input_embedding_tensor.unsqueeze(0)  # Add batch dimension

    # Handle empty input embedding tensor
    if input_embedding_tensor.nelement() == 0:
        return []

    # Resize input embedding tensor if dimensionality doesn't match
    if input_embedding_tensor.size(-1) != vault_embeddings.size(-1):
        try:
            input_embedding_tensor=torch.nn.functional.interpolate(input_embedding_tensor, size=vault_embeddings.size(-1), mode='nearest')
        except Exception as e:
            print(f"Error resizing input embedding tensor: {e}")
            return []

    # Ensure vault embeddings have the same dimensionality as input embedding tensor
    if input_embedding_tensor.size(-1) != vault_embeddings.size(-1):
        raise ValueError("Dimensionality of input embedding and vault embeddings must match.")

    cos_scores=torch.nn.functional.cosine_similarity(input_embedding_tensor, vault_embeddings, dim=-1)
    top_k=min(top_k, len(cos_scores))
    top_indices=torch.topk(cos_scores, k=top_k)[1].tolist()
    relevant_context=[vault_content[idx].strip() for idx in top_indices]
    return relevant_context





def rewrite_query(user_input_json, conversation_history, ollama_model, client):
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

    response=client.post(
        f"http://localhost:11434/v1/completions",
        json={
            "model": ollama_model,
            "prompt": prompt,
            "max_tokens": 200,
        },
        timeout=60.0
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

# Function to handle the chat logic with Ollama
def ollama_chat(user_input, system_message, vault_embeddings, vault_content, ollama_model, conversation_history, client):
    conversation_history.append({"role": "user", "content": user_input})

    if len(conversation_history) > 1:
        query_json={
            "Query": user_input,
            "Rewritten Query": ""
        }
        rewritten_query_json=rewrite_query(json.dumps(query_json), conversation_history, ollama_model, client)
        if rewritten_query_json:
            rewritten_query_data=json.loads(rewritten_query_json)
            rewritten_query=rewritten_query_data["Rewritten Query"]
            print(f"Original Query: {user_input}")
            print(f"Rewritten Query: {rewritten_query}")
        else:
            print("Failed to rewrite the query.")
            return None
    else:
        rewritten_query=user_input

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
        timeout=60.0
    )

    if response.status_code == 200:
        content=response.json()
        assistant_response=content["choices"][0]["message"]["content"]
        conversation_history.append({"role": "assistant", "content": assistant_response})

        # Write assistant response to a file
        with open("assistant_response.txt", "w", encoding="utf-8") as response_file:
            response_file.write(assistant_response + "\n")

        return assistant_response
    else:
        print(f"HTTP request failed with status code {response.status_code}.")
        return None

def main():
    parser=argparse.ArgumentParser(description="Ollama Chat")
    parser.add_argument("--model", default="dolphin-llama3:latest", help="Ollama model to use (default: llama3)")
    args=parser.parse_args()

    client=httpx.Client(base_url='http://localhost:11434/v1', timeout=60.0)

    vault_content=[]
    vault_file_path="vault.txt"
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
            print("Assistant response:\n" + response)

if __name__ == "__main__":
    main()

