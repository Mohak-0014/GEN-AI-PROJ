from flask import Flask, request, render_template
from transformers import T5Tokenizer, T5ForConditionalGeneration

app = Flask(__name__)

model = T5ForConditionalGeneration.from_pretrained("./t5-code-explainer")
tokenizer = T5Tokenizer.from_pretrained("./t5-code-explainer")

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/explain', methods=['POST'])
def explain():
    code = request.form['code']
    input_text = f"explain the following code in plain English:\n{code}"
    input_ids = tokenizer.encode(input_text, return_tensors="pt", truncation=True, max_length=256)
    output_ids = model.generate(
        input_ids,
        max_length=500,
        num_beams=4,  # Beam search (better quality)
        no_repeat_ngram_size=3,  # Prevent repeating trigrams
        early_stopping=True,  # Stop when best output is found
        repetition_penalty=1.2  # Penalize repetition
    )

    explanation = tokenizer.decode(output_ids[0], skip_special_tokens=True)
    return render_template('index.html', code=code, explanation=explanation)

if __name__ == '__main__':
    app.run(debug=True)