import base64
import os
import uuid

import cv2
import gradio as gr
import numpy as np
import requests


import numpy as np
import gradio as gr

IMAGE_CACHE_DIRECTORY = "data"
API_URL = "https://api.openai.com/v1/chat/completions"


def preprocess_image(image: np.ndarray) -> np.ndarray:
    image = np.fliplr(image)
    return cv2.cvtColor(image, cv2.COLOR_RGB2BGR)


def encode_image_to_base64(image: np.ndarray) -> str:
    success, buffer = cv2.imencode(".jpg", image)
    if not success:
        raise ValueError("Could not encode image to JPEG format.")

    encoded_image = base64.b64encode(buffer).decode("utf-8")
    return encoded_image


def compose_payload(image: np.ndarray, prompt: str) -> dict:
    base64_image = encode_image_to_base64(image)
    return {
        "model": "gpt-4-vision-preview",
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt},
                    {
                        "type": "image_url",
                        "image_url": {"url": f"data:image/jpeg;base64,{base64_image}"},
                    },
                ],
            }
        ],
        "max_tokens": 300,
    }


def compose_headers(api_key: str) -> dict:
    return {"Content-Type": "application/json", "Authorization": f"Bearer {api_key}"}


def prompt_image(api_key: str, image: np.ndarray, prompt: str) -> str:
    headers = compose_headers(api_key=api_key)
    payload = compose_payload(image=image, prompt=prompt)
    response = requests.post(url=API_URL, headers=headers, json=payload).json()

    if "error" in response:
        raise ValueError(response["error"]["message"])
    return response["choices"][0]["message"]["content"]


def cache_image(image: np.ndarray) -> str:
    image_filename = f"{uuid.uuid4()}.jpeg"
    os.makedirs(IMAGE_CACHE_DIRECTORY, exist_ok=True)
    image_path = os.path.join(IMAGE_CACHE_DIRECTORY, image_filename)
    cv2.imwrite(image_path, image)
    return image_path


def get_random_chord() -> str:
    # Add as many chords you want
    chords = ["A", "B", "C", "D", "E", "F", "G"]
    return np.random.choice(chords)


def respond(api_key: str, image: np.ndarray, chord_value: str, chat_history):
    if not api_key:
        raise ValueError("API_KEY is not set.")

    prompt = f"Is this a {chord_value} chord?"

    image = preprocess_image(image=image)
    cached_image_path = cache_image(image)
    response = prompt_image(api_key=api_key, image=image, prompt=prompt)

    new_chord = get_random_chord()

    chat_history.append(((cached_image_path,), None))
    chat_history.append((prompt, response))
    chat_history.append((None, f"Now let's practice *{new_chord}* chord!"))

    return chat_history, new_chord


with gr.Blocks() as demo:
    with gr.Row():
        webcam = gr.Image(source="webcam", streaming=True)
        with gr.Column():
            api_key_textbox = gr.Textbox(label="OpenAI API KEY", type="password")
            chord = gr.Textbox(label="chord", value="A", type="text", visible=False)
            chatbot = gr.Chatbot(
                height=500,
                bubble_full_width=False,
                value=[
                    (None, "Welcome to the GuitarGPT!"),
                    (None, f"Let's practice *{chord.value}* chord!"),
                ],
            )
            btn = gr.Button(value="Send answer")
            btn.click(
                fn=respond,
                inputs=[api_key_textbox, webcam, chord, chatbot],
                outputs=[chatbot, chord],
            )

demo.launch(debug=False, show_error=True)
