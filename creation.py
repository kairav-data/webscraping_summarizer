import requests
from bs4 import BeautifulSoup
import torch
from transformers import pipeline


# Function to fetch and parse HTML content
def fetch_html(url):
    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36'}
    response = requests.get(url, headers=headers)
    return BeautifulSoup(response.content, 'html.parser')


# Function to extract main text content from HTML
def extract_text(soup):
    # Here we assume the main content is within <p> tags
    paragraphs = soup.find_all('p', class_='pw-post-body-paragraph')
    text = ' '.join([p.get_text() for p in paragraphs])
    return text


# Function to preprocess text (you can add more preprocessing steps if needed)
def preprocess_text(text):
    # Here we just strip extra whitespace
    return ' '.join(text.split())


# Function to summarize text using Hugging Face's Transformers
def summarize_text(text, max_length=524):
    summarizer = pipeline('summarization', model='sshleifer/distilbart-cnn-12-6', revision="a4f8f3e",framework="pt")
    summary = summarizer(text, max_length=max_length, min_length=30, do_sample=False)
    return summary[0]['summary_text']


# Main function to summarize web content
def summarize_web_content(url):
    # Fetch and parse HTML content
    soup = fetch_html(url)

    # Extract and preprocess text content
    text = extract_text(soup)
    preprocessed_text = preprocess_text(text)

    # Summarize the preprocessed text
    summary = summarize_text(preprocessed_text)

    return summary


# Example usage
url = "https://medium.com/@hughmcguire/why-can-t-we-read-anymore-503c38c131fe"
summary = summarize_web_content(url)
print(summary)
