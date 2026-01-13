"""
Voice-Text Converter App
A Streamlit application for bidirectional audio-text conversion using Hugging Face models.
Supports Voice-to-Text (Speech-to-Text) and Text-to-Voice (Text-to-Speech).
"""

import streamlit as st
import torch
import torchaudio
import soundfile as sf
import librosa
import numpy as np
import tempfile
import os
from transformers import (
    AutoModelForSpeechSeq2Seq,
    AutoProcessor,
    pipeline,
    SpeechT5Processor,
    SpeechT5ForTextToSpeech,
    SpeechT5HifiGan
)
from datasets import load_dataset

# ===========================
# Configuration
# ===========================
MAX_AUDIO_SIZE_MB = 10
MAX_TEXT_LENGTH = 1000
SAMPLE_RATE = 16000

# ===========================
# Model Loading Functions
# ===========================

@st.cache_resource
def load_stt_model():
    """
    Load Speech-to-Text model (Whisper Tiny) with caching.
    Returns a Hugging Face pipeline for automatic speech recognition.
    """
    try:
        device = "cuda" if torch.cuda.is_available() else "cpu"
        torch_dtype = torch.float16 if torch.cuda.is_available() else torch.float32
        
        model_id = "openai/whisper-tiny"
        
        model = AutoModelForSpeechSeq2Seq.from_pretrained(
            model_id,
            torch_dtype=torch_dtype,
            low_cpu_mem_usage=True,
            use_safetensors=True
        )
        model.to(device)
        
        processor = AutoProcessor.from_pretrained(model_id)
        
        pipe = pipeline(
            "automatic-speech-recognition",
            model=model,
            tokenizer=processor.tokenizer,
            feature_extractor=processor.feature_extractor,
            max_new_tokens=128,
            torch_dtype=torch_dtype,
            device=device,
        )
        
        return pipe
    except Exception as e:
        st.error(f"Failed to load Speech-to-Text model: {str(e)}")
        st.info("Please check your internet connection and try again.")
        return None


@st.cache_resource
def load_tts_model():
    """
    Load Text-to-Speech model (SpeechT5) with vocoder and caching.
    Returns processor, model, and vocoder for speech synthesis.
    """
    try:
        processor = SpeechT5Processor.from_pretrained("microsoft/speecht5_tts")
        model = SpeechT5ForTextToSpeech.from_pretrained("microsoft/speecht5_tts")
        vocoder = SpeechT5HifiGan.from_pretrained("microsoft/speecht5_hifigan")
        
        # Load speaker embeddings from a dataset
        embeddings_dataset = load_dataset("Matthijs/cmu-arctic-xvectors", split="validation")
        speaker_embeddings = torch.tensor(embeddings_dataset[7306]["xvector"]).unsqueeze(0)
        
        return processor, model, vocoder, speaker_embeddings
    except Exception as e:
        st.error(f"Failed to load Text-to-Speech model: {str(e)}")
        st.info("Please check your internet connection and try again.")
        return None, None, None, None


# ===========================
# Audio Processing Functions
# ===========================

def process_audio_file(audio_file):
    """
    Process uploaded audio file and resample to 16kHz if needed.
    Returns audio array and sample rate.
    """
    try:
        # Save uploaded file temporarily
        with tempfile.NamedTemporaryFile(delete=False, suffix='.wav') as tmp_file:
            tmp_file.write(audio_file.read())
            tmp_path = tmp_file.name
        
        # Load audio using librosa (handles multiple formats)
        audio, sr = librosa.load(tmp_path, sr=SAMPLE_RATE)
        
        # Clean up temporary file
        os.unlink(tmp_path)
        
        return audio, sr
    except Exception as e:
        st.error(f"Error processing audio file: {str(e)}")
        return None, None


def transcribe_audio(audio_file, pipe):
    """
    Transcribe audio file to text using the STT pipeline.
    Returns transcribed text.
    """
    try:
        # Process audio file
        audio, sr = process_audio_file(audio_file)
        if audio is None:
            return None
        
        # Perform transcription
        with st.spinner("Transcribing audio..."):
            result = pipe(audio)
            transcribed_text = result["text"]
        
        return transcribed_text
    except Exception as e:
        st.error(f"Transcription failed: {str(e)}")
        return None


def generate_tts(text, processor, model, vocoder, speaker_embeddings):
    """
    Generate speech from text using the TTS model.
    Returns path to generated audio file.
    """
    try:
        # Process text input
        inputs = processor(text=text, return_tensors="pt")
        
        # Generate speech
        with st.spinner("Generating audio..."):
            with torch.no_grad():
                speech = model.generate_speech(
                    inputs["input_ids"],
                    speaker_embeddings,
                    vocoder=vocoder
                )
        
        # Save audio to temporary file
        output_path = tempfile.mktemp(suffix=".wav")
        sf.write(output_path, speech.numpy(), samplerate=16000)
        
        return output_path
    except Exception as e:
        st.error(f"Audio generation failed: {str(e)}")
        return None


# ===========================
# Main Application
# ===========================

def main():
    # Page configuration
    st.set_page_config(
        page_title="Voice-Text Converter",
        page_icon="🎙️",
        layout="centered"
    )
    
    # Title and description
    st.title("🎙️ Voice-Text Converter App")
    st.markdown("""
    Convert between voice and text seamlessly using AI models.
    - **Voice to Text**: Upload audio files and get text transcriptions
    - **Text to Voice**: Enter text and generate natural speech
    """)
    
    # Sidebar with information
    with st.sidebar:
        st.header("ℹ️ Information")
        st.markdown("""
        **Supported Audio Formats:**
        - WAV, MP3, OGG
        
        **Limitations:**
        - Max audio size: 10MB
        - Max text length: 1000 characters
        
        **Models Used:**
        - STT: OpenAI Whisper Tiny
        - TTS: Microsoft SpeechT5
        """)
        
        st.markdown("---")
        st.markdown("**Note:** Models will download on first use and are cached for subsequent runs.")
    
    # Create tabs for different functionalities
    tab1, tab2 = st.tabs(["🎤 Voice to Text", "🔊 Text to Voice"])
    
    # ===========================
    # Voice to Text Tab
    # ===========================
    with tab1:
        st.header("Voice to Text (Speech-to-Text)")
        st.markdown("Upload an audio file to transcribe it to text.")
        
        # File uploader
        audio_file = st.file_uploader(
            "Upload Audio File",
            type=['wav', 'mp3', 'ogg'],
            help="Supported formats: WAV, MP3, OGG (Max size: 10MB)"
        )
        
        # Check file size
        if audio_file is not None:
            file_size_mb = len(audio_file.getvalue()) / (1024 * 1024)
            if file_size_mb > MAX_AUDIO_SIZE_MB:
                st.error(f"File size ({file_size_mb:.2f} MB) exceeds the maximum limit of {MAX_AUDIO_SIZE_MB} MB.")
                audio_file = None
        
        # Transcribe button
        if st.button("Transcribe", key="transcribe_btn", disabled=(audio_file is None)):
            # Load STT model
            stt_pipe = load_stt_model()
            
            if stt_pipe is not None:
                # Transcribe audio
                transcribed_text = transcribe_audio(audio_file, stt_pipe)
                
                if transcribed_text:
                    st.success("Transcription completed!")
                    
                    # Display transcribed text
                    st.text_area(
                        "Transcribed Text",
                        value=transcribed_text,
                        height=200,
                        key="transcribed_output"
                    )
                    
                    # Download button for text
                    st.download_button(
                        label="📥 Download Transcription",
                        data=transcribed_text,
                        file_name="transcribed_text.txt",
                        mime="text/plain"
                    )
    
    # ===========================
    # Text to Voice Tab
    # ===========================
    with tab2:
        st.header("Text to Voice (Text-to-Speech)")
        st.markdown("Enter text and generate natural-sounding speech.")
        
        # Text input
        input_text = st.text_area(
            "Enter Text to Convert",
            height=150,
            max_chars=MAX_TEXT_LENGTH,
            help=f"Maximum {MAX_TEXT_LENGTH} characters"
        )
        
        # Character count
        char_count = len(input_text)
        st.caption(f"Characters: {char_count}/{MAX_TEXT_LENGTH}")
        
        # Validate input
        is_valid_input = len(input_text.strip()) > 0
        
        if not is_valid_input and char_count > 0:
            st.warning("Please enter some text to convert.")
        
        # Generate button
        if st.button("Generate Audio", key="generate_btn", disabled=(not is_valid_input)):
            # Load TTS model
            processor, model, vocoder, speaker_embeddings = load_tts_model()
            
            if all([processor, model, vocoder, speaker_embeddings is not None]):
                # Generate audio
                audio_path = generate_tts(
                    input_text,
                    processor,
                    model,
                    vocoder,
                    speaker_embeddings
                )
                
                if audio_path:
                    st.success("Audio generated successfully!")
                    
                    # Play audio
                    with open(audio_path, 'rb') as audio_file:
                        audio_bytes = audio_file.read()
                        st.audio(audio_bytes, format='audio/wav')
                        
                        # Download button for audio
                        st.download_button(
                            label="📥 Download Audio",
                            data=audio_bytes,
                            file_name="generated_audio.wav",
                            mime="audio/wav"
                        )
                    
                    # Clean up temporary file
                    try:
                        os.unlink(audio_path)
                    except:
                        pass


if __name__ == "__main__":
    main()
