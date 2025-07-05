# 🧠 AI Code Explainer using Generative AI

## 🔍 Overview

The **AI Code Explainer** is an intelligent web-based tool that explains Python code in plain English using a fine-tuned **T5 (Text-to-Text Transfer Transformer)** model. It's designed to support programming beginners, educators, and developers who want a better understanding of unfamiliar code.

## 🚀 Features

- Code-to-text generation using a fine-tuned T5 model.
- Clean and interactive web interface built with Flask.
- Supports natural-language explanations for a wide range of Python functions.
- Fast and lightweight — runs on CPU with <2s response time.

## 🛠️ Tech Stack

- **Model:** T5-small from Hugging Face Transformers
- **Languages:** Python
- **Libraries:** PyTorch, Transformers, Datasets, SentencePiece
- **Frontend/Backend:** Flask + HTML/CSS (Jinja templating)
- **Tokenizer:** SentencePiece

## 📦 Project Structure

```
├── train.py             # Script to fine-tune the T5 model
├── app.py               # Flask web server
├── templates/
│   └── index.html       # Frontend web UI
├── static/              # (Optional) CSS/JS assets
├── model/               # Saved model and tokenizer
└── requirements.txt     # Project dependencies
```

## 🧪 Testing

### ✅ Testing Strategy

The system was tested to ensure:
- Explanation relevance
- Tolerance to syntax variations
- Robustness to incomplete code

### 🧷 Testing Types

- **Unit Testing** – Tokenization, preprocessing, and generation modules.
- **Integration Testing** – End-to-end functionality from input to UI output.
- **User Testing** – Usability and feedback from real users.
- **Performance Testing** – Average response time measured on CPU.

### 🧪 Results

| Metric             | Result                   |
|--------------------|--------------------------|
| Average Response   | < 2 seconds (CPU)        |
| Accuracy           | High for standard code   |
| Limitations        | Struggles with obfuscated or highly complex code |

## 🌱 Future Enhancements

- 🧠 Fine-tune on more languages (e.g., JavaScript, C++) using CodeSearchNet.
- 🔁 Add user feedback loop for continual improvement.
- 🧩 Line-by-line explanation mode.
- 🌐 Real-time extensions for Chrome or VS Code.


