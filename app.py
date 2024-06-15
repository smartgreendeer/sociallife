from flask import Flask, render_template, jsonify, request
from src.helper import load_pdf, text_split, download_hugging_face_embeddings
from langchain_community.llms import CTransformers
from transformers import AutoModel, AutoTokenizer
from dotenv import load_dotenv
import torch, os, chromadb
import time
from llama_index.llms.huggingface import HuggingFaceLLM
from transformers import AutoModelForCausalLM
from langchain_community.chat_message_histories import ChatMessageHistory
from llama_index.core.prompts.prompts import SimpleInputPrompt
from transformers import pipeline



app = Flask(__name__)

# Load environment variables
load_dotenv()

#the tokens and keys
PINECONE_API_KEY = os.environ.get('PINECONE_API_KEY')
PINECONE_API_ENV = os.environ.get('PINECONE_API_ENV')
HF_TOKEN = os.environ.get("HF_TOKEN")

# Load data and embeddings
extracted_data = load_pdf("data/")
text_chunks = text_split(extracted_data)

#download embeddings
embeddings = download_hugging_face_embeddings()

system_prompt = """
You are a Question and Answer assistant. Your goal is to answer questions based on the provided context and instructions. Your answers should be grammatically correct and easily understandable.

Context: {context}
Question: {question}

Helpful answer:
"""
assistant_prompt = "{query_str}"

query_wrapper_prompt = SimpleInputPrompt(assistant_prompt)

model_config = {'protected_namespaces': ()}

try:
    # Setup CTransformers LLM
    llm = CTransformers(
        generator = pipeline("text-generation", model="https://huggingface.co/meta-llama/Llama-2-7b-chat-hf/blob/main/config.json"),
        tokenizer = AutoTokenizer.from_pretrained,
        auth_token=HF_TOKEN,
        model_type="llama",
        config={'max_new_tokens': 256, 'temperature': 0.3, 'top_k': 20},
        force_download=True,
        **model_config# Adjusted for performance
    )
except EnvironmentError as e:
    print(f"Error loading model: {e}")
    model = None

    


# Flask routes
@app.route("/")
def index():
    return render_template('bot.html')

@app.route("/get", methods=["GET", "POST"])
def chat():
    try:
        msg = request.form["msg"]
        input_text = msg
        print(f"Received message: {input_text}")
        
        # Display spinner
        result = {"generated_text": "Thinking..."}
        
        # Simulate processing delay
        time.sleep(1)
        if llm:
            # Retrieve response from the model
            result = llm.generate([input_text])
            print(f"LLMResult: {result}")
        
            # Access the generated text from the result object
            if result.generations and result.generations[0]:
                generated_text = result.generations[0][0].text
            else:
                generated_text = "No response generated."
            
            print(f"Response: {generated_text}")
            
            return str(generated_text)
        
    except Exception as e:
        print(f"Error: {e}")
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    app.run(host="0.0.0.0", port=8080, debug=True)