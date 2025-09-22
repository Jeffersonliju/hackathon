import os
from flask import Flask, render_template, request, jsonify
import google.generativeai as genai

# Configure the Gemini API
# Make sure to set your API key as an environment variable
# export GOOGLE_API_KEY="your_api_key"
try:
    genai.configure(api_key=os.environ["GOOGLE_API_KEY"])
except KeyError:
    print("Please set the GOOGLE_API_KEY environment variable.")
    exit()


# Create the model
generation_config = {
    "temperature": 1,
    "top_p": 0.95,
    "top_k": 64,
    "max_output_tokens": 8192,
    "response_mime_type": "text/plain",
}

model = genai.GenerativeModel(
    model_name="gemini-1.5-flash",
    generation_config=generation_config,
    system_instruction="You are a helpful AI assistant for disease prediction. Your role is to provide information and suggestions based on the symptoms described by the user. You are not a substitute for professional medical advice. Always advise users to consult with a healthcare professional for an accurate diagnosis and treatment plan.",
)

app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/chat', methods=['POST'])
def chat():
    message = request.json['message']

    chat_session = model.start_chat(
        history=[
        ]
    )

    response = chat_session.send_message(message)

    return jsonify({'response': response.text})

if __name__ == '__main__':
    app.run(debug=True)
