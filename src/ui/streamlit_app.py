import streamlit as st
import requests

st.title("GenAI Post Generator")
st.write("Generate health-related posts using our fine-tuned GPT-2 model.")

# Input fields for prompt, max_length, and temperature
prompt = st.text_input("Enter your prompt:")
max_length = st.slider("Max Length", min_value=50, max_value=200, value=100)
temperature = st.slider("Temperature", min_value=0.5, max_value=1.0, value=0.7)

if st.button("Generate Post"):
    if not prompt:
        st.error("Please enter a prompt.")
    else:
        # Prepare the payload
        payload = {
            "prompt": prompt,
            "max_length": max_length,
            "temperature": temperature
        }
        # Call the API (assuming it's running locally on port 8000)
        try:
            response = requests.post("http://localhost:8080/generate", json=payload)
            response.raise_for_status()
            data = response.json()
            generated_post = data.get("generated_post", "")
            st.subheader("Generated Post:")
            st.write(generated_post)
        except requests.exceptions.RequestException as e:
            st.error(f"Error generating post: {e}")
