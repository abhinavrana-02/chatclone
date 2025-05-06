import streamlit as st
from transformers import GPT2LMHeadModel, GPT2Tokenizer
import torch

# ✅ Page configuration must come before any other Streamlit elements
st.set_page_config(page_title="GPT-2 Text Generator", layout="wide")

# ✅ Load GPT-2 model and tokenizer
@st.cache_resource
def load_model():
    tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
    model = GPT2LMHeadModel.from_pretrained("gpt2")
    model.eval()
    return tokenizer, model

tokenizer, model = load_model()

# ✅ App UI
st.title("🧠 GPT-2 Local Text Generator")
st.markdown("Enter a prompt and generate text locally using GPT-2 (124M).")

prompt = st.text_area("Prompt", "Once upon a time in India...")

max_length = st.slider("Max Output Length", min_value=50, max_value=300, value=100)
temperature = st.slider("Creativity (Temperature)", 0.1, 1.5, 0.7)
top_p = st.slider("Top-p Sampling (Nucleus)", 0.1, 1.0, 0.9)

if st.button("Generate"):
    with st.spinner("Generating..."):
        inputs = tokenizer(prompt, return_tensors="pt")
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_length=max_length,
                temperature=temperature,
                top_p=top_p,
                do_sample=True
            )
        result = tokenizer.decode(outputs[0], skip_special_tokens=True)
        st.subheader("Generated Text")
        st.write(result)
