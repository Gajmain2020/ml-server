from flask import Flask, request, jsonify
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import torch
from flask_cors import CORS

app = Flask(__name__)
CORS(app)  # Allow CORS for all origins

model_path = "./saved_model"  # Path to your locally saved model

# Load model and tokenizer
try:
    tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=True)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_path)
    print("✅ Model loaded successfully from:", model_path)
except Exception as e:
    print("❌ Error loading model:", e)
    raise e

@app.route("/")
def home():
    return "Model is running."

@app.route("/correct", methods=["POST"])
def correct_text():
    try:
        data = request.get_json()
        input_text = data.get("text", "")
        if not input_text:
            return jsonify({"error": "No text provided"}), 400

        # Tokenize input text
        inputs = tokenizer(input_text, return_tensors="pt", truncation=True, padding=True)

        # Generate output
        with torch.no_grad():
            outputs = model.generate(**inputs, num_beams=4, max_length=512, early_stopping=True)

        corrected_text = tokenizer.decode(outputs[0], skip_special_tokens=True)

        # Determine error type (basic logic)
        if input_text.strip().lower() == corrected_text.strip().lower():
            error_type = "No Error"
        elif "?" in input_text and "?" not in corrected_text:
            error_type = "Punctuation"
        elif any(char.isupper() for char in corrected_text) and not any(char.isupper() for char in input_text):
            error_type = "Capitalization"
        else:
            error_type = "Grammar / Syntax"

        return jsonify({
            "original_text": input_text,
            "corrected": corrected_text,
            "error_type": error_type
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    app.run(debug=True)
