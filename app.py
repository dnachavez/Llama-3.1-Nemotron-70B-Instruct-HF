import streamlit as st
from transformers import pipeline, AutoTokenizer, AutoModelForCausalLM
import torch

@st.cache_resource
def load_model():
    # Load tokenizer and model
    tokenizer = AutoTokenizer.from_pretrained("nvidia/Llama-3.1-Nemotron-70B-Instruct-HF")
    model = AutoModelForCausalLM.from_pretrained(
        "nvidia/Llama-3.1-Nemotron-70B-Instruct-HF",
        device_map="auto",
        torch_dtype=torch.float16,
    )
    return tokenizer, model

def generate_response(prompt, tokenizer, model):
    # Encode the prompt
    input_ids = tokenizer.encode(prompt, return_tensors='pt').to(model.device)
    
    # Generate a response
    with torch.no_grad():
        output_ids = model.generate(
            input_ids,
            max_length=1024,
            do_sample=True,
            top_p=0.95,
            temperature=0.8,
            num_return_sequences=1,
        )
    # Decode the response
    response = tokenizer.decode(output_ids[0], skip_special_tokens=True)
    return response

def main():
    st.title("Llama-3.1-Nemotron-70B-Instruct-HF Chatbot")
    st.write("Interact with the Llama-3.1-Nemotron-70B-Instruct-HF model.")

    # Load the model and tokenizer
    with st.spinner("Loading the model..."):
        tokenizer, model = load_model()

    # Initialize conversation history
    if 'messages' not in st.session_state:
        st.session_state.messages = []

    # Display conversation history
    for message in st.session_state.messages:
        if message['role'] == 'user':
            st.markdown(f"**You:** {message['content']}")
        else:
            st.markdown(f"**Bot:** {message['content']}")

    # User input
    user_input = st.text_input("You:", key="input")

    if user_input:
        # Append user message to conversation history
        st.session_state.messages.append({"role": "user", "content": user_input})
        
        # Generate and display the bot's response
        with st.spinner("Generating response..."):
            response = generate_response(user_input, tokenizer, model)
        st.session_state.messages.append({"role": "bot", "content": response})
        
        # Rerun the app to display the updated conversation
        st.experimental_rerun()

if __name__ == "__main__":
    main()
