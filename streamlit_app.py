# python -m streamlit run streamlit_app.py
# https://unimate-chatbot.streamlit.app/

import streamlit as st
import nltk
import random
from streamlit_extras.colored_header import colored_header
from nltk.tokenize import sent_tokenize
from transformers import AutoModelWithLMHead, AutoTokenizer
from transformers import pipeline
from embeddings import find_context
from EmbeddingsUsingPineCone.pinecone_embeddings import get_similar_docs


#======================Streamlit Application =================================

st.set_page_config(page_title="UniMate - Your University Mate",page_icon="💬")
st.title('UniMate Chatbot 💬🤖')

user_selected_model = st.selectbox(
    "Select a model",
    ("DialoGPT","GPT-2")
)

colored_header(label='', description='', color_name='blue-100')

# Configure the side bar
with st.sidebar:
    st.markdown('''                  
        ## About
       The UniMate chatbot application aims to answer queries posed by the Sunway University/College community, particularly knowlegeable about the information regarding Student Handbook. 
                            
        ## Specifications
        This chatbot is built using:
        - Fine-tuned [GPT-2 model](https://huggingface.co/gpt2)
        - Fine-tuned [DialoGPT](https://huggingface.co/microsoft/DialoGPT-small)
        - Streamlit
        
        💡 Note: No API key required!
        
        ### Connect with me ( ´ ∀ `)
    ''')

    #Configure the layout of the social media icons
    column1, column2 = st.columns(2)
    column1.markdown("[![LinkedIn](<https://cdn2.iconfinder.com/data/icons/social-media-2285/512/1_Linkedin_unofficial_colored_svg-48.png>)](<https://www.linkedin.com/in/yujia-lim-b85081213/>)")
    column2.markdown("[![GitHub](<https://img.icons8.com/material-outlined/48/000000/github.png>)](<https://github.com/jia02>)")

#========================== Configure the model =================================

# Define remote directory to the fine-tuned Huggingface Space's DialoGPT model
dialogpt_model_path = "YJia/dialogpt-test"
@st.cache_resource
def load_dialogpt_model(path):
    nltk.download('punkt')  
    dialogpt_model = AutoModelWithLMHead.from_pretrained(path)
    dialogpt_tokenizer = AutoTokenizer.from_pretrained('microsoft/DialoGPT-small', padding_side='left')
    return dialogpt_model, dialogpt_tokenizer

dialogpt_model, dialogpt_tokenizer = load_dialogpt_model(dialogpt_model_path)

# Define remote directory to the fine-tuned Huggingface Space's GPT-2 model
gpt2QA_model_path = "YJia/gptQA-test"
@st.cache_resource
def load_gpt2QA_model(path):
    gpt2QA_pipeline = pipeline(
        "question-answering", 
        model=path, 
        tokenizer=path, 
        max_answer_len=50,
        top_k=3
    )
    return gpt2QA_pipeline

gpt2QA_question_answerer = load_gpt2QA_model(gpt2QA_model_path)

#======================Configure the generation of response =============================

# Function to extract complete sentence from the generated response
def is_complete_sentence(sentence):
    # Check if a sentence ends with a period, question mark, or exclamation mark
    return sentence.endswith('.') or sentence.endswith('?') or sentence.endswith('!')

# Function for generating LLM response from DialoGPT model
def query(payload, maxLength, model, tokenizer): 
    bot_input_ids = tokenizer.encode(payload["inputs"]["text"] + tokenizer.eos_token, return_tensors='pt')

    chat_history_ids = model.generate(
        bot_input_ids, 
        max_length=maxLength,
        pad_token_id=dialogpt_tokenizer.eos_token_id,  
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

# Function for generating LLM response from GPT-2 model
def gpt2QA_generate_response(prompt):

    # Using local embeddings
    #context = find_context(prompt) #from using local embeddings.py 

    # Using langchain's embeddings retrived from Pinecone vector database
    context = get_similar_docs(prompt) 

    result = gpt2QA_question_answerer(question=prompt, context=context)

    answers = []
    for example in result:
        answer = example['answer'].strip()
        if (answer not in ["i", ".", "Add/Drop period", "the Subject Add/Drop period.", "students"]):
            answers.append(answer)
            print(answers)

    # Choose a random answer from the list
    selected_answer = random.choice(answers)

    return selected_answer

#==================================== Load the messages =================================

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
    response = None  # Initialize the response variable

    with st.chat_message("assistant"):
        with st.spinner("Generating a response..."):
            try:
                if user_selected_model == "DialoGPT":
                    st.success("Switched to DialoGPT model")
                    # Load the response generated by the fine-tuned dialogpt model
                    dialogpt_response = query( {"inputs": {"text": prompt}}, 150, dialogpt_model, dialogpt_tokenizer)
                    st.write(dialogpt_response["generated_text"]) 
                    response = dialogpt_response["generated_text"]

                else: #user_selected_model == "GPT-2"
                    # Load the response generated by the fine-tuned GPT-2 model
                    st.success("Switched to GPT-2 model")
                    gpt2_response = gpt2QA_generate_response(prompt)
                    st.write(gpt2_response) 
                    response = gpt2_response

            except Exception as e:
                st.error(f"An error occurred during response generation: {str(e)}")
                # Update the chat history with the error message
                st.session_state.messages.append({"role": "assistant", "content": f"An error occurred: {str(e)}"})
                response = "I don't understand what you are asking. Please ask a complete question."

    # Now you can use the response variable in the rest of your code
    message = {"role": "assistant", "content": response}
    st.session_state.messages.append(message)


