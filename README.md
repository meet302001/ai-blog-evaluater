
# AI Blog Evaluator (OpenAI & Gemini)

This Streamlit web app allows you to generate blog content using **OpenAI GPT-3.5** and **Google Gemini 1.5 Flash**, then compares both outputs using:

- Readability scores (Flesch-Kincaid, Gunning Fog, SMOG, etc.)
- Lexical diversity
- Topic relevance (via semantic similarity)
- Sentiment analysis (DistilBERT)
- Hallucination detection using LLM-based fact checking

---

## Features

- Blog generation using OpenAI & Gemini models
- Evaluation using NLP metrics
- LLM-based fact checking to flag hallucinations
- Side-by-side comparison interface with Streamlit

---

## Installation

```bash
git clone https://github.com/meet302001/ai-blog-evaluater.git
cd ai-blog-evaluater
pip install -r requirements.txt
```

---

## Setup `.env`

Create a `.env` file in the root directory and add your API keys:

```env
OPENAI_API_KEY=your_openai_api_key
GEMINI_API_KEY=your_gemini_api_key
```

You can get:
- OpenAI key from: https://platform.openai.com/account/api-keys
- Gemini key from: https://makersuite.google.com/app/apikey

---

## Run the App

```bash
streamlit run main2.py
```

---

## Tech Stack

- Streamlit for UI
- OpenAI GPT-3.5 Turbo
- Google Gemini 1.5 Flash
- Sentence Transformers (`MiniLM`)
- Hugging Face Transformers (DistilBERT sentiment)
- TextStat for readability scoring


---

## Author

**Meet Bhanushali**  
📧 bhanushallimeet302001@gmail.com
🔗 [LinkedIn](https://linkedin.com/in/bhanushallimeet)

---

## ⭐️ Support

If you find this project helpful, please consider giving it a ⭐ on GitHub!
