import requests
import os
from bs4 import BeautifulSoup
import torch
from transformers import pipeline

# Set environment variables to control TensorFlow behavior and suppress warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # Suppress TensorFlow logging (0 = all logs, 1 = info, 2 = warning, 3 = error)
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'  # Disable oneDNN custom operations

# Set environment variable to suppress symlink warning from huggingface
os.environ['HF_HUB_DISABLE_SYMLINKS_WARNING'] = '1'


# Function to fetch and parse HTML content
def fetch_html(url):
    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36'}
    response = requests.get(url, headers=headers)
    print("Fetched HTML content successfully....... ")
    return BeautifulSoup(response.content, 'html.parser')


# Function to extract main text content from HTML
def extract_text(soup):
    paragraphs = soup.find_all('p', class_='pw-post-body-paragraph')
    text = ' '.join([p.get_text() for p in paragraphs])

    return text


# Function to preprocess text
def preprocess_text(text):
    # Here we just strip extra whitespace
    print("preprocessed text successfully........... ")
    return ' '.join(text.split())


# Function to split the text into smaller chunks if it's too long
def split_text(text, max_length=1024):
    words = text.split()
    chunks = []
    current_chunk = []
    current_length = 0
    for word in words:
        current_length += len(word) + 1  # +1 for the space
        if current_length > max_length:
            chunks.append(" ".join(current_chunk))
            current_chunk = [word]
            current_length = len(word) + 1
        else:
            current_chunk.append(word)
    if current_chunk:
        chunks.append(" ".join(current_chunk))

    return chunks


# Function to summarize text using Hugging Face's Transformers
def summarize_text(text, max_length=1024, batch_size=4):
    # Split the text into smaller chunks
    text_chunks = split_text(text, max_length=max_length)

    # Initialize the summarization pipeline with GPU support if available
    device = 0 if torch.cuda.is_available() else -1
    summarizer = pipeline("summarization", model="facebook/bart-large-cnn", device=device)

    # Summarize each chunk in batches
    summaries = []
    for i in range(0, len(text_chunks), batch_size):
        batch = text_chunks[i:i + batch_size]
        batch_summaries = summarizer(batch, max_length=150, min_length=30, do_sample=False)
        summaries.extend([summary['summary_text'] for summary in batch_summaries])

    final_summary = " ".join(summaries)
    print("Summary :")
    return final_summary


# Main function to summarize web content
def summarize_web_content(url):
    # Fetch and parse HTML content
    soup = fetch_html(url)

    # Extract and preprocess text content
    text = extract_text(soup)
    preprocessed_text = preprocess_text(text)
    with open("preprocessed.txt", "w", encoding="utf-8") as file:
        file.write(preprocessed_text)
    # Summarize the preprocessed text
    summary = summarize_text(preprocessed_text)

    return summary


# Example usage
url = "https://medium.com/@nottremi/i-lost-my-voice-as-the-only-thing-i-learned-while-growing-up-is-to-stay-silent-e76a5e75ed9a"
summary = summarize_web_content(url)

with open("summary.txt", "w", encoding="utf-8") as file:
    file.write(summary)

print(summary)
