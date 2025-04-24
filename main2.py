import streamlit as st
import openai
import google.generativeai as genai
import os
from dotenv import load_dotenv
import torch
import textstat
from sentence_transformers import SentenceTransformer, util
from collections import Counter
from transformers import AutoTokenizer, AutoModelForSequenceClassification, pipeline

# Load environment variables
load_dotenv()

# Set API keys
OPENAI_API_KEY = ""
GEMINI_API_KEY = ""

# Set OpenAI API key
openai.api_key = OPENAI_API_KEY

# Set up Gemini configuration
genai.configure(api_key=GEMINI_API_KEY)

def get_sentiment_pipeline():
    model_name = "distilbert-base-uncased-finetuned-sst-2-english"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(model_name)
    return pipeline("sentiment-analysis", model=model, tokenizer=tokenizer, device=-1)

# Load models and tools
sentiment_pipeline = get_sentiment_pipeline()
embedding_model = SentenceTransformer("all-MiniLM-L6-v2")

# Blog generation

def generate_blog(model_choice, prompt):
    if model_choice == "openai":
        response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": "You are a helpful assistant that writes blog posts."},
                {"role": "user", "content": prompt}
            ],
            max_tokens=500,
            temperature=0.7
        )
        return response.choices[0].message['content'].strip()

    elif model_choice == "gemini":
        model = genai.GenerativeModel("models/gemini-1.5-flash")
        response = model.generate_content(prompt)
        return response.text

    return "Invalid model choice."

# Metrics

def readability_scores(text):
    return {
        "Flesch Reading Ease": textstat.flesch_reading_ease(text),
        "Flesch-Kincaid Grade": textstat.flesch_kincaid_grade(text),
        "Gunning Fog Index": textstat.gunning_fog(text),
        "SMOG Index": textstat.smog_index(text),
        "Dale-Chall Score": textstat.dale_chall_readability_score(text)
    }

def lexical_diversity(text):
    words = text.split()
    return len(set(words)) / len(words)

def topic_relevance(prompt, blog_text):
    embeddings = embedding_model.encode([prompt, blog_text], convert_to_tensor=True)
    similarity = util.pytorch_cos_sim(embeddings[0], embeddings[1])
    return float(similarity.item())

def get_sentiment(text):
    return sentiment_pipeline(text[:512])[0]  # Limit to 512 tokens

# Hallucination detection

def fact_check_sentence(sentence):
    check_prompt = f"""
Fact check the following statement. Respond with only one word: True, False, or Unverifiable.

"{sentence}"
"""
    result = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=[{"role": "user", "content": check_prompt}],
        max_tokens=5,
        temperature=0
    )
    return result.choices[0].message['content'].strip()

def detect_hallucinations(blog_text):
    sentences = blog_text.split(".")
    results = []
    for sentence in sentences:
        sentence = sentence.strip()
        if sentence:
            verdict = fact_check_sentence(sentence)
            results.append((sentence, verdict))
    return results

# Streamlit app

st.title("🤖 AI Blog Generator and Evaluator")
prompt_input = st.text_area("Enter your blog prompt:", height=150)

if st.button("Generate and Evaluate") and prompt_input:
    with st.spinner("Generating blog content..."):
        blog_openai = generate_blog("openai", prompt_input)
        blog_gemini = generate_blog("gemini", prompt_input)

    col1, col2 = st.columns(2)

    with col1:
        st.header("🤖 OpenAI Blog")
        st.write(blog_openai)

        st.subheader("📊 Readability")
        st.json(readability_scores(blog_openai))

        st.subheader("🔤 Lexical Diversity")
        st.write(lexical_diversity(blog_openai))

        st.subheader("📎 Topic Relevance")
        st.write(topic_relevance(prompt_input, blog_openai))

        st.subheader("❤️ Sentiment")
        st.json(get_sentiment(blog_openai))

        st.subheader("🧠 Hallucination Detection")
        openai_hallucinations = detect_hallucinations(blog_openai)
        for sentence, verdict in openai_hallucinations:
            flag = "✅" if verdict == "True" else "❌" if verdict == "False" else "⚠️"
            st.markdown(f"{flag} **[{verdict}]** {sentence}")

    with col2:
        st.header("🤖 Gemini Blog")
        st.write(blog_gemini)

        st.subheader("📊 Readability")
        st.json(readability_scores(blog_gemini))

        st.subheader("🔤 Lexical Diversity")
        st.write(lexical_diversity(blog_gemini))

        st.subheader("📎 Topic Relevance")
        st.write(topic_relevance(prompt_input, blog_gemini))

        st.subheader("❤️ Sentiment")
        st.json(get_sentiment(blog_gemini))

        st.subheader("🧠 Hallucination Detection")
        gemini_hallucinations = detect_hallucinations(blog_gemini)
        for sentence, verdict in gemini_hallucinations:
            flag = "✅" if verdict == "True" else "❌" if verdict == "False" else "⚠️"
            st.markdown(f"{flag} **[{verdict}]** {sentence}")