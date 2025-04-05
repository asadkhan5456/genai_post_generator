from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import os
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

# Load fine-tuned GPT-2 model and tokenizer
try:
    model = GPT2LMHeadModel.from_pretrained(model_path)
    tokenizer = GPT2Tokenizer.from_pretrained(model_path)
except Exception as e:
    raise RuntimeError(f"Error loading model or tokenizer: {e}")

# Ensure model is in evaluation mode
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
    
    # Clean the prompt
    prompt = clean_text(request.prompt)
    inputs = tokenizer.encode(prompt, return_tensors="pt")
    
    # Generate text using the fine-tuned model
    outputs = model.generate(
        inputs,
        max_length=request.max_length + 20,  # increase max length a bit
        temperature=request.temperature,     
        do_sample=True,
        top_k=40,       # try lowering top_k slightly
        top_p=0.90,     # slightly adjust top_p
        num_return_sequences=1,
)
    
    generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return {"generated_post": generated_text}

@app.get("/")
def read_root():
    return {"message": "Welcome to the GenAI Post Generator API. Use the /generate endpoint to create health-related posts."}
