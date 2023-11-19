from flask import Flask, render_template, request, send_file
from transformers import AutoProcessor, MusicgenForConditionalGeneration
from IPython.display import Audio
import torch
import scipy
import os
import json



app = Flask(__name__)


#check if gpu is available 
#TRY COMMENTING OUT

def query_gpt(artist):
    import openai
    openai.api_type = "azure"
    openai.api_key = 'f8590d0a2d77474d80246e2231565a49'

    openai.api_base = 'https://api.umgpt.umich.edu/azure-openai-api/ptu'
    openai.api_version = '2023-03-15-preview'

    try:
        response = openai.ChatCompletion.create(
            engine='gpt-4',
            messages=[
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": f"I am trying to get a one line description for {artist}'s vibe of music. Can you please give me that one line, in the format of the following sentence: '80s blues track with groovy saxophone'?"}
            ]
        )

        # print the response
        print(response['choices'][0]['message']['content'])
        return response['choices'][0]['message']['content']
    # except:
    #     print("error with API")
    #     return ""

    except openai.error.APIError as e:
        # Handle API error here, e.g. retry or log
        print(f"OpenAI API returned an API Error: {e}")
        return ""

    except openai.error.AuthenticationError as e:
        # Handle Authentication error here, e.g. invalid API key
        print(f"OpenAI API returned an Authentication Error: {e}")
        return ""

    except openai.error.APIConnectionError as e:
        # Handle connection error here
        print(f"Failed to connect to OpenAI API: {e}")
        return ""

    except openai.error.InvalidRequestError as e:
        # Handle connection error here
        print(f"Invalid Request Error: {e}")
        return ""

    except openai.error.RateLimitError as e:
        # Handle rate limit error
        print(f"OpenAI API request exceeded rate limit: {e}")
        return ""

    except openai.error.ServiceUnavailableError as e:
        # Handle Service Unavailable error
        print(f"Service Unavailable: {e}")
        return ""

    except openai.error.Timeout as e:
        # Handle request timeout
        print(f"Request timed out: {e}")
        return ""

    except:
        # Handles all other exceptions
        print("An exception has occured.")
        return ""


def generate_audio(inp):
    global model
    processor = AutoProcessor.from_pretrained("facebook/musicgen-small")
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

    scipy.io.wavfile.write("/Users/prakhar/Desktop/music-ai-generator/static/musicgen_out.wav", rate=sampling_rate, data=audio_values[0, 0].cpu().numpy())

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/process_input', methods=['POST'])
def process_input():
    user_input = request.form['user_input']
    artist_input = request.form['artist_input']
    print("user: ", user_input)
    print("artist: ", artist_input)

    s = ""
    if (user_input != ""):
        s += user_input + ", "
    if (artist_input != ""):
        res = query_gpt(artist_input)
        if (res != ""):
            print("prob with query")
            s += res


    # Do something with the user input (e.g., store it in a variable, process it, etc.)
    print(s)
    if (s != ""):
        generate_audio(s)
    else:
        print("invalid input")
    audio_file_path = "/Users/prakhar/Desktop/music-ai-generator/static/musicgen_out.wav"
    if os.path.exists(audio_file_path):
        # Send the audio file as a response
        return render_template('index.html', audio_file=audio_file_path)
    else:
        # Return an error response if the file is not found
        return "File not found", 404

if __name__ == '__main__':
    model = MusicgenForConditionalGeneration.from_pretrained("facebook/musicgen-small")
    app.run(debug=True)
