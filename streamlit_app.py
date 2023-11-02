# https://unimate-2023.streamlit.app/
# python -m streamlit run streamlit_app.py
# pip install -r requirements.txt

import streamlit as st
import re
import torch
#import nltk
from streamlit_extras.colored_header import colored_header
from streamlit_extras.add_vertical_space import add_vertical_space
from nltk.tokenize import sent_tokenize
from transformers import GPT2Tokenizer, GPT2LMHeadModel, AutoModelWithLMHead, AutoTokenizer

#nltk.download('punkt')

# Define remote directory to the Huggingface Space's GPT model
gpt_model_path = "YJia/gpt2-test" 
# Load the fine-tuned model and tokenizer
gpt_model = GPT2LMHeadModel.from_pretrained(gpt_model_path)
gpt_tokenizer = GPT2Tokenizer.from_pretrained(gpt_model_path)

# Define remote directory to the Huggingface Space's dialo-GPT model
model_name = "YJia/dialogpt-test"  
model = AutoModelWithLMHead.from_pretrained(model_name)
tokenizer = AutoTokenizer.from_pretrained('microsoft/DialoGPT-small', padding_side='left')


# Function for generating LLM response from gpt2 model
def gpt2_generate_response(model, tokenizer, prompt, max_length):
    input_ids = tokenizer.encode(prompt, return_tensors="pt")

    # Create the attention mask and pad token id
    attention_mask = torch.ones_like(input_ids)
    pad_token_id = tokenizer.eos_token_id

    # Configure the hyperparameters when generating response
    output = model.generate(
        input_ids,
        max_length=max_length,
        num_return_sequences=1,
        attention_mask=attention_mask,
        pad_token_id=pad_token_id,
        do_sample=True,
        temperature=0.3,
        top_k=20,
        top_p=0.2,
        repetition_penalty=1.0
    ) 

    # Convert the output to a string
    response = tokenizer.decode(output[0])

    # Use regular expressions to extract text within double quotes
    matches = re.findall(r'"text": "([^"]*)"', response)

    if matches:
        generated_text = matches[0]

    else:
        generated_text = response  # Use the full response if "text" field not found

    return generated_text

# Function for generating LLM response from dialogpt model
def is_complete_sentence(sentence):
    # Check if a sentence ends with a period, question mark, or exclamation mark
    return sentence.endswith('.') or sentence.endswith('?') or sentence.endswith('!')

def query(payload, maxLength):
    bot_input_ids = tokenizer.encode(payload["inputs"]["text"] + tokenizer.eos_token, return_tensors='pt')

    chat_history_ids = model.generate(
        bot_input_ids, 
        max_length=maxLength,
        pad_token_id=tokenizer.eos_token_id,  
        no_repeat_ngram_size=4,       
        do_sample=False, 
        top_k=20,  # Adjust the value of top_k
        top_p=0.2,  # Adjust the value of top_p
        temperature=0.3  # Adjust the value of temperature
    )
    
    # Decode the generated response and tokenize into sentences
    generated_response = tokenizer.decode(chat_history_ids[:, bot_input_ids.shape[-1]:][0], skip_special_tokens=True)
    sentences = sent_tokenize(generated_response)

    # Filter out incomplete sentences
    complete_sentences = [sentence.strip() for sentence in sentences if is_complete_sentence(sentence)]

    # Join the complete sentences to form the final response
    extracted_response = ' '.join(complete_sentences)
        
    return {"generated_text": extracted_response}

#==================================== Streamlit Application =================================
st.set_page_config(page_title="UniMate : Your SU Mate",page_icon="💬")
st.title('🤖💬 UniMate Chatbot 💬')

user_selected_model = st.selectbox(
    "Select a model",
    ("DialoGPT-small", "gpt-2")
)

colored_header(label='', description='', color_name='blue-100')

# Configure the side bar
with st.sidebar:
    st.title('💬 UniMate : Your University Mate')
    st.markdown('''
    ## About
    This chatbot application aims to provide academic advising to the Sunway University's community, particularly knwoledgeable about the information regarding Student Handbook. 
                           
    ## Specifications
    This chatbot is built using:
    - Fine-tuned [GPT-2 model](https://huggingface.co/gpt2)
    - Fine-tuned [microsoft/DialoGPT-small](https://huggingface.co/microsoft/DialoGPT-small)
    - Streamlit
    
    💡 Note: No API key required!
    ''')
    add_vertical_space(4)
    st.write('Made by [LYJ](<https://github.com/Jia02>)')

# Store LLM generated responses
if "messages" not in st.session_state.keys():
    st.session_state.messages = [{
        "role": "assistant", 
        "content": "Ayeeee! Thanks for using me! How may I help you?"
    }]

# Display chat messages
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.write(message["content"])

# Configure the layout of input/response containers
chat_container = st.container()

# User-provided prompt
with chat_container:
    if prompt := st.chat_input("Enter your question here"):
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.write(prompt)

# Generate a new response if last message is not from assistant
if st.session_state.messages[-1]["role"] != "assistant":
    with st.chat_message("assistant"):
        with st.spinner("Generating a response..."):
            try:
                if user_selected_model == "gpt-2": 
                    # Load the response generated by the fine-tuned gpt2 model
                    gpt2_response = gpt2_generate_response(gpt_model, gpt_tokenizer, prompt, 200)
                    st.write(gpt2_response) 
                    response = gpt2_response
                    
                else:
                    # Load the response generated by the fine-tuned dialogpt model
                    dialogpt_response = query( {"inputs": {"text": prompt}}, 200)
                    st.write(dialogpt_response["generated_text"]) 
                    response = dialogpt_response["generated_text"]

            except Exception as e:
                    st.error(f"An error occurred during response generation: {str(e)}")
                    # Update the chat history with the error message
                    st.session_state.messages.append({"role": "assistant", "content": f"An error occurred: {str(e)}"})
            message = {"role": "assistant", "content": response}
            st.session_state.messages.append(message) 
