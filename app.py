from flask import Flask, render_template, request, send_file
from transformers import AutoProcessor, MusicgenForConditionalGeneration
from IPython.display import Audio
import torch
import scipy
import os


app = Flask(__name__)


#check if gpu is available 
#TRY COMMENTING OUT



def generate_audio(inp):
    processor = AutoProcessor.from_pretrained("facebook/musicgen-small")
    model = MusicgenForConditionalGeneration.from_pretrained("facebook/musicgen-small")
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    model.to(device)
    sampling_rate = model.config.audio_encoder.sampling_rate
    inputs = processor(
        # text=["80s pop track with bassy drums and synth", "90s rock song with loud guitars and heavy drums"],
        text=inp,
        padding=True,
        return_tensors="pt",
    )
    audio_values = model.generate(**inputs.to(device), do_sample=True, guidance_scale=3, max_new_tokens=256)

    scipy.io.wavfile.write("/home/avibhattacharya/Desktop/mhacks/musicgen_out.wav", rate=sampling_rate, data=audio_values[0, 0].cpu().numpy())

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/process_input', methods=['POST'])
def process_input():
    user_input = request.form['user_input']
    # Do something with the user input (e.g., store it in a variable, process it, etc.)
    print(f"User input: {user_input}")
    generate_audio(user_input)
    audio_file_path = "/home/avibhattacharya/Desktop/mhacks/static/musicgen_out.wav"
    if os.path.exists(audio_file_path):
        # Send the audio file as a response
        return render_template('index.html', audio_file=audio_file_path)
    else:
        # Return an error response if the file is not found
        return "File not found", 404

if __name__ == '__main__':
    app.run(debug=True)
