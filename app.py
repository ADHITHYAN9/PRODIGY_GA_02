from flask import Flask, request, jsonify, send_from_directory, render_template
from flask_cors import CORS
from authtoken import auth_token
import torch
from torch import autocast
from diffusers import StableDiffusionPipeline
import os

app = Flask(__name__, static_folder="static", template_folder="templates")
CORS(app)


modelid = "CompVis/stable-diffusion-v1-4"
device = "cuda" if torch.cuda.is_available() else "cpu"
pipe = StableDiffusionPipeline.from_pretrained(
    modelid,
    revision="fp16",
    torch_dtype=torch.float16 if device == "cuda" else torch.float32,
    use_auth_token=auth_token
)
pipe.to(device)


os.makedirs("static", exist_ok=True)

@app.route("/")
def home():
    return render_template("index.html")  

@app.route("/generate", methods=["POST"])
def generate():
    data = request.get_json()
    prompt = data.get("prompt")
    if not prompt:
        return jsonify({"error": "Prompt is required"}), 400

   
    with autocast(device):
        result = pipe(prompt, guidance_scale=8.5)
        image = result.images[0]

   
    filename = "generated_image.png"
    filepath = os.path.join("static", filename)
    image.save(filepath)

    return jsonify({"filename": filename})

@app.route("/static/<path:filename>")
def static_files(filename):
    return send_from_directory("static", filename)

if __name__ == "__main__":
    app.run(debug=True)
