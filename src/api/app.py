import os
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import re
import torch
from transformers import GPT2LMHeadModel, GPT2Tokenizer

app = FastAPI(title="GenAI Post Generator API")

class PostRequest(BaseModel):
    prompt: str
    max_length: int = 100
    temperature: float = 0.7

# Determine the project root by moving up three directories from this file's location
project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
model_path = os.path.join(project_root, "model", "genai_post_model")

# Load model and tokenizer: if the local model directory exists, use it;
# otherwise, load the base GPT2 model from Hugging Face Hub.
try:
    if os.path.exists(model_path):
        model = GPT2LMHeadModel.from_pretrained(model_path, local_files_only=True)
        tokenizer = GPT2Tokenizer.from_pretrained(model_path, local_files_only=True)
    else:
        # For CI/CD and testing, fallback to the base model
        model = GPT2LMHeadModel.from_pretrained("gpt2")
        tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
except Exception as e:
    raise RuntimeError(f"Error loading model or tokenizer: {e}")

model.eval()

def clean_text(text: str) -> str:
    """Clean the input text by lowercasing and removing extra whitespace."""
    text = text.lower()
    text = re.sub(r'\s+', ' ', text).strip()
    return text

@app.post("/generate")
def generate_post(request: PostRequest):
    if not request.prompt:
        raise HTTPException(status_code=400, detail="Prompt is required.")
    
    prompt = clean_text(request.prompt)
    inputs = tokenizer.encode(prompt, return_tensors="pt")
    
    outputs = model.generate(
        inputs,
        max_length=request.max_length,
        temperature=request.temperature,
        do_sample=True,
        top_k=50,
        top_p=0.95,
    )
    
    generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return {"generated_post": generated_text}

@app.get("/")
def read_root():
    return {"message": "Welcome to the GenAI Post Generator API. Use the /generate endpoint to create health-related posts."}
