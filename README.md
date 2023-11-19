# tune.ai

## Introduction

Our platform, **tune.ai**, is an AI-driven music generator capable of creating music based on a specific vibe and/or an artist's style. It's a perfect tool for musicians seeking inspiration, businesses developing unique tones for their brand, podcasters, filmmakers, or anyone who loves music.

We rigorously expanded on Meta Audiocraft's MusicGen model to generate tunes which align with the given vibe. We added an option to choose an artist's style through UM-GPT's API. The user-friendly web interface was created using HTML/CSS/JS for front-end design and Flask for back-end functionality.

## Installation

### Requirements
pip install transformers
pip install flask
pip install openai==0.28

### Usage

1. Clone the repository.
2. In the file app.py, insert your OpenAI API key in lines 18, and update the lines 94 and 123 with the appropriate directory where the project is located locally.
3. Run **python3 app.py**
4. Open the generated link in a browser and enjoy tune.ai!
