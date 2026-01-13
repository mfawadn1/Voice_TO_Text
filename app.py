import streamlit as st
import torch
from transformers import pipeline, AutoProcessor, AutoModelForTextToWaveform
import soundfile as sf
import numpy as np
from io import BytesIO
import time
from audio_recorder_streamlit import audio_recorder

# Page configuration
st.set_page_config(
    page_title="Voice-to-Text & Text-to-Voice",
    page_icon="🎙️",
    layout="wide"
)

# Initialize session state
if 'transcribed_text' not in st.session_state:
    st.session_state.transcribed_text = ""
if 'generated_audio' not in st.session_state:
    st.session_state.generated_audio = None
if 'recorded_audio' not in st.session_state:
    st.session_state.recorded_audio = None

@st.cache_resource
def load_speech_to_text_model():
    """Load and cache the Whisper model for speech-to-text."""
    try:
        with st.spinner("Loading speech-to-text model (first time only)..."):
            # Using Whisper tiny for faster performance
            pipe = pipeline(
                "automatic-speech-recognition",
                model="openai/whisper-tiny",
                device=-1  # CPU only
            )
        return pipe
    except Exception as e:
        st.error(f"Error loading speech-to-text model: {e}")
        return None

@st.cache_resource
def load_text_to_speech_model():
    """Load and cache the TTS model for text-to-speech."""
    try:
        with st.spinner("Loading text-to-speech model (first time only)..."):
            # Using Facebook's MMS TTS model
            processor = AutoProcessor.from_pretrained("facebook/mms-tts-eng")
            model = AutoModelForTextToWaveform.from_pretrained("facebook/mms-tts-eng")
        return processor, model
    except Exception as e:
        st.error(f"Error loading text-to-speech model: {e}")
        return None, None

def transcribe_audio(audio_bytes, stt_model):
    """Transcribe audio bytes to text."""
    try:
        start_time = time.time()
        
        # Save audio to temporary bytes
        with st.spinner("Transcribing audio..."):
            # The audio_recorder returns WAV format
            result = stt_model(audio_bytes)
            transcription = result["text"]
        
        elapsed_time = time.time() - start_time
        st.success(f"Transcription complete! (Time: {elapsed_time:.2f}s)")
        
        return transcription
    except Exception as e:
        st.error(f"Transcription error: {e}")
        return None

def generate_speech(text, processor, model):
    """Generate speech from text."""
    try:
        start_time = time.time()
        
        with st.spinner("Generating speech..."):
            # Process text
            inputs = processor(text=text, return_tensors="pt")
            
            # Generate speech
            with torch.no_grad():
                output = model(**inputs).waveform
            
            # Convert to numpy array
            audio_data = output.squeeze().cpu().numpy()
            
            # Normalize audio
            audio_data = audio_data / np.max(np.abs(audio_data))
            
        elapsed_time = time.time() - start_time
        st.success(f"Speech generation complete! (Time: {elapsed_time:.2f}s)")
        
        return audio_data, model.config.sampling_rate
    except Exception as e:
        st.error(f"Speech generation error: {e}")
        return None, None

def audio_to_bytes(audio_data, sample_rate):
    """Convert audio numpy array to bytes."""
    buffer = BytesIO()
    sf.write(buffer, audio_data, sample_rate, format='WAV')
    buffer.seek(0)
    return buffer.getvalue()

# Main UI
st.title("🎙️ Voice-to-Text & Text-to-Voice Application")
st.markdown("Convert speech to text and text to speech using free, open-source AI models.")

# Sidebar
with st.sidebar:
    st.header("⚙️ Settings")
    st.info("**Models Used:**\n- Speech-to-Text: Whisper Tiny\n- Text-to-Speech: Facebook MMS TTS")
    
    st.markdown("---")
    st.markdown("### 📋 Instructions")
    st.markdown("""
    **Voice-to-Text:**
    1. Click the microphone to record
    2. Speak clearly
    3. Click again to stop
    4. View transcription below
    
    **Text-to-Voice:**
    1. Enter or paste text
    2. Click 'Generate Speech'
    3. Listen to the audio
    4. Download if needed
    """)

# Create two columns for the main functionality
col1, col2 = st.columns(2)

# ===== VOICE-TO-TEXT SECTION =====
with col1:
    st.header("🎤 Voice-to-Text")
    st.markdown("Record your voice and get the text transcription.")
    
    # Audio recorder
    audio_bytes = audio_recorder(
        text="Click to record",
        recording_color="#e74c3c",
        neutral_color="#3498db",
        icon_name="microphone",
        icon_size="2x"
    )
    
    if audio_bytes:
        st.session_state.recorded_audio = audio_bytes
        st.audio(audio_bytes, format="audio/wav")
        
        # Transcribe button
        if st.button("🔄 Transcribe Audio", key="transcribe_btn"):
            stt_model = load_speech_to_text_model()
            if stt_model:
                transcription = transcribe_audio(audio_bytes, stt_model)
                if transcription:
                    st.session_state.transcribed_text = transcription
    
    # Display transcription
    if st.session_state.transcribed_text:
        st.markdown("### 📝 Transcription:")
        transcribed = st.text_area(
            "Edit transcription if needed:",
            value=st.session_state.transcribed_text,
            height=150,
            key="transcription_display"
        )
        
        # Download buttons
        col_dl1, col_dl2 = st.columns(2)
        with col_dl1:
            st.download_button(
                label="⬇️ Download Text",
                data=transcribed,
                file_name="transcription.txt",
                mime="text/plain"
            )
        with col_dl2:
            if st.session_state.recorded_audio:
                st.download_button(
                    label="⬇️ Download Audio",
                    data=st.session_state.recorded_audio,
                    file_name="recording.wav",
                    mime="audio/wav"
                )
        
        if st.button("🗑️ Clear Transcription", key="clear_transcription"):
            st.session_state.transcribed_text = ""
            st.session_state.recorded_audio = None
            st.rerun()

# ===== TEXT-TO-VOICE SECTION =====
with col2:
    st.header("🔊 Text-to-Voice")
    st.markdown("Enter text and convert it to speech.")
    
    # Text input
    text_input = st.text_area(
        "Enter text to convert to speech:",
        height=150,
        placeholder="Type your text here...",
        help="Enter the text you want to convert to speech"
    )
    
    # Generate speech button
    if st.button("🎵 Generate Speech", key="generate_btn", disabled=not text_input):
        if text_input.strip():
            processor, model = load_text_to_speech_model()
            if processor and model:
                audio_data, sample_rate = generate_speech(text_input, processor, model)
                if audio_data is not None:
                    st.session_state.generated_audio = {
                        'data': audio_data,
                        'rate': sample_rate
                    }
        else:
            st.warning("Please enter some text first!")
    
    # Display generated audio
    if st.session_state.generated_audio:
        st.markdown("### 🎧 Generated Audio:")
        audio_bytes = audio_to_bytes(
            st.session_state.generated_audio['data'],
            st.session_state.generated_audio['rate']
        )
        st.audio(audio_bytes, format="audio/wav")
        
        # Download button
        st.download_button(
            label="⬇️ Download Audio",
            data=audio_bytes,
            file_name="generated_speech.wav",
            mime="audio/wav"
        )
        
        if st.button("🗑️ Clear Audio", key="clear_audio"):
            st.session_state.generated_audio = None
            st.rerun()

# Footer
st.markdown("---")
st.markdown(
    """
    <div style='text-align: center'>
        <p>Built with Streamlit | Models: OpenAI Whisper & Facebook MMS TTS</p>
        <p style='font-size: 0.8em; color: gray;'>All processing is done locally - no data is stored</p>
    </div>
    """,
    unsafe_allow_html=True
)
