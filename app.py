import streamlit as st
from PIL import Image
from transformers import pipeline
import torch
import io
from io import StringIO
import os
import shutil
import wavio

# Audio parameters
samplerate = 44100
duration = 1
frequency = 440



def predict_img(img_path):
    # Use a pipeline as a high-level helper
    img_pipe = pipeline("image-classification", model="Emiel/cub-200-bird-classifier-swin")
    ## Running the inference
    result = img_pipe(img_path)[0]
    ## Printing the result label
    return result['label'], img_path

def predict_audio(audio_path):
    # Use a pipeline as a high-level helper
    audio_pipe = pipeline("audio-classification", model="saadashraf/birds_model") #DBD-research-group/Wav2Vec2-Base-BirdSet-XCL sgoedecke/wav2vec2_birdcalls JamesStratford/ast-finetuned-voice-of-birds
    ## Running the inference
    result = audio_pipe(audio_path)[0]
    ## Printing the result label
    return result['label']

if __name__ == "__main__":
    with st.sidebar:
        st.title('Birdie:bird:')
        st.subheader('A Bird Classifier', divider="grey")
        choice = st.radio("Navigation",["Image Classification","Audio Classification","Chat with Birdie"])
        st.info("This app allows you to classify birds based on images, songs, and Q&A with a chatbot.")

    if choice == "Image Classification":
        st.title("By Sight :eye:")
        col1, col2 = st.columns(2)
        with col1:
            file = st.file_uploader(label="Upload An Image for Classification")
            if file:
                bytes_data = file.getvalue()
                image = Image.open(io.BytesIO(bytes_data))
                st.image(image)
                image.save('img.png')

        with col2:
            try:
                prediction = predict_img('img.png')
                prediction = prediction[0].replace('_', ' ')
                st.title(f"Birdie detected a {prediction}")
            except Exception as e:
                pass



    if choice == "Audio Classification":
        st.title("By Ear :ear:")
        col1, col2 = st.columns(2)
        with col1:
            file = st.file_uploader(label="Upload an Audio file for Classification")
            output_filename = "audio.wav"
            if file:
                st.audio(file)
                wavio.write(output_filename, bytearray(file.getvalue()), samplerate, sampwidth=2)

        with col2:
            try:
                prediction = predict_audio('audio.wav')
                # prediction = prediction[0].replace('_', ' ')
                st.title(f"Birdie detected a {  prediction}")
            except Exception as e:
                pass
                # st.info(e)

    if choice == "Chat with Birdie":
        st.title("Birdie :small[A bird identifing chat bot]")

        # initialize chat history
        if "messages" not in st.session_state:
            st.session_state.messages = []

        # display chat messages from history on app rerun
        for message in st.session_state.messages:
            with st.chat_message(message["role"]):
                st.markdown(message["content"])

        # handle user input
        if prompt := st.chat_input("What is up?"):
            # display user message
            st.chat_message("user").markdown(prompt)
            # add message to history
            st.session_state.messages.append({"role": "user", "content": prompt})

            response = f"Echo: {prompt}"
            # display assistant response
            with st.chat_message("assistant"):
                st.markdown(response)
            # add response to history
            st.session_state.messages.append({"role": "assistant", "content": response})
