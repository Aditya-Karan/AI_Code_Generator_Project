from flask import Flask, render_template, request, jsonify
from transformers import GPT2LMHeadModel, GPT2Tokenizer
import os

app = Flask(__name__)

# Load model
model_path = os.path.join("auto_coding", "model", "gpt2_medium_fine_tuned_coder")
model = GPT2LMHeadModel.from_pretrained(model_path, local_files_only=True)
tokenizer = GPT2Tokenizer.from_pretrained(model_path, local_files_only=True)
model.config.pad_token_id = tokenizer.eos_token_id
model.eval()

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/generate', methods=['POST'])
def generate():
    data = request.json
    language = data.get("language", "python").lower()
    context = data.get("context", "")

    prompt = f"<{language}> {context}"
    input_ids = tokenizer.encode(prompt, return_tensors="pt")
    attention_mask = input_ids.ne(tokenizer.pad_token_id)

    outputs = model.generate(
        input_ids=input_ids,
        attention_mask=attention_mask,
        max_length=128,
        temperature=0.7,
        num_return_sequences=1
    )

    decoded = tokenizer.decode(outputs[0], skip_special_tokens=True)
    trimmed = decoded.split("\n\n")[0]

    return jsonify({"output": trimmed})

if __name__ == '__main__':
    app.run(debug=True)
