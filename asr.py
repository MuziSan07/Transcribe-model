from huggingface_hub import InferenceClient
from dotenv import load_dotenv
import os

# Load environment variables from .env file
load_dotenv()

# Get the token from environment variable
hf_token = os.getenv("HUGGING_FACE_API")

# Initialize the client with the token
client = InferenceClient(token=hf_token)

def transcribe_audio(file_path: str) -> str:
    output = client.automatic_speech_recognition(
        file_path,
        model="boumehdi/wav2vec2-large-xlsr-moroccan-darija"
    )
    return output.get("text", "")
