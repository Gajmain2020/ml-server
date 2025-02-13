import pickle
from flask import Flask, request, jsonify
from transformers import T5Tokenizer

app = Flask(__name__)

# Load the model from model.pkl
try:
    with open("model.pkl", "rb") as f:
        model = pickle.load(f)  # Load trained model

    tokenizer = T5Tokenizer.from_pretrained("t5-base")  # Load tokenizer
    print("✅ Model loaded successfully from model.pkl!")
except Exception as e:
    print(f"❌ Error loading model: {e}")

def detect_error_type(original, corrected):
    """
    Compares original and corrected sentences to determine the type of error.
    """
    if original.lower() == corrected.lower():
        return "No errors detected"
    
    original_words = set(original.lower().split())
    corrected_words = set(corrected.lower().split())

    if len(original_words) < len(corrected_words):
        return "Missing words"
    elif len(original_words) > len(corrected_words):
        return "Extra words"
    else:
        return "Grammar structure issue"

@app.route('/')
def home():
    return "Grammar Correction API is running!"

@app.route('/correct', methods=['POST'])
def predict():
    try:
        data = request.json  # Get JSON input
        sentence = data.get("sentence", "").strip()

        if not sentence:
            return jsonify({"error": "No sentence provided"}), 400

        # Preprocess input
        input_text = f"grammar: {sentence}"
        input_ids = tokenizer.encode(input_text, return_tensors="pt")

        # Generate corrected text
        output_ids = model.generate(input_ids, max_length=100, num_beams=5, early_stopping=True)
        corrected_sentence = tokenizer.decode(output_ids[0], skip_special_tokens=True).replace("Grammar: ", "")

        # Detect error type
        error_type = detect_error_type(sentence, corrected_sentence)

        return jsonify({"original": sentence, "corrected": corrected_sentence, "error_type": error_type})

    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True)
