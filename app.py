import streamlit as st
from transformers import pipeline

@st.cache_resource
def load_pipeline():
    # Load the text-generation pipeline
    pipe = pipeline(
        "text-generation",
        model="nvidia/Llama-3.1-Nemotron-70B-Instruct-HF",
        device_map="auto",
    )
    return pipe

def main():
    st.title("Llama-3.1-Nemotron-70B-Instruct-HF Chatbot")
    st.write("Interact with the Llama-3.1-Nemotron-70B-Instruct-HF model using the pipeline.")

    # Load the pipeline
    with st.spinner("Loading the model..."):
        pipe = load_pipeline()

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

        # Prepare the input for the pipeline
        conversation = "\n".join([f"{msg['role']}: {msg['content']}" for msg in st.session_state.messages])

        # Generate and display the bot's response
        with st.spinner("Generating response..."):
            output = pipe(
                conversation,
                max_length=1024,
                do_sample=True,
                top_p=0.95,
                temperature=0.8,
                num_return_sequences=1,
            )[0]['generated_text']

            # Extract the new part of the response
            response = output[len(conversation):].strip()

        st.session_state.messages.append({"role": "bot", "content": response})

        # Rerun the app to display the updated conversation
        st.experimental_rerun()

if __name__ == "__main__":
    main()
