from flask import Flask, render_template, request, jsonify
import requests
from PyPDF2 import PdfReader
import os
import nltk
import re
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer
import numpy as np
from transformers import T5Tokenizer, T5ForConditionalGeneration
from rouge_score import rouge_scorer
import textstat
import sqlite3
import textwrap
import torch
import io
import tempfile
from sklearn.metrics.pairwise import cosine_similarity
import networkx as nx
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from sentence_transformers import SentenceTransformer

app = Flask(__name__)

# Initialize NLTK data directory
nltk_data_path = os.path.join(tempfile.gettempdir(), 'nltk_data')
os.makedirs(nltk_data_path, exist_ok=True)
os.environ['NLTK_DATA'] = nltk_data_path
nltk.data.path.append(nltk_data_path)

# Download required NLTK resources
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt', download_dir=nltk_data_path)

try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('stopwords', download_dir=nltk_data_path)


class TextCleaner:
    """Cleans input text by removing unnecessary elements"""

    def __init__(self):
        self.header_footer_patterns = [
            r'^[0-9]+$',  # Page numbers
            r'^[A-Z][A-Z\s]+$',  # ALL CAPS HEADERS
            r'^[^\w\s]*$',  # Special characters only
            r'^\s*$',  # Empty lines
            r'^[^\w]*[0-9]+[^\w]*$',  # Lines with only numbers
            r'^[^\w]*(?:references|bibliography|appendix)[^\w]*$',  # References section
            r'^©.*$',  # Copyright notices
            r'^https?://\S+$',  # URLs
            r'^[\w\.-]+@[\w\.-]+\.\w+$'  # Email addresses
        ]

    def clean_text(self, text):
        """Apply cleaning rules to input text"""
        lines = text.split('\n')
        cleaned_lines = []

        for line in lines:
            if not any(re.match(pattern, line.strip().lower()) for pattern in self.header_footer_patterns):
                cleaned_lines.append(line)

        cleaned_text = '\n'.join(cleaned_lines)
        cleaned_text = re.sub(r'\s+', ' ', cleaned_text).strip()
        return cleaned_text


class Summarizer:
    """Handles extractive and abstractive summarization"""

    def __init__(self):
        # Initialize T5 model
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.tokenizer = T5Tokenizer.from_pretrained('t5-base')
        self.model = T5ForConditionalGeneration.from_pretrained('t5-base').to(self.device)
        # Initialize TF-IDF vectorizer
        self.vectorizer = TfidfVectorizer(stop_words=stopwords.words('english'))

    def extractive_summarize(self, text, num_sentences=5):
        """Improved extractive summary with sentence fusion"""
        sentences = sent_tokenize(text)
        if len(sentences) <= num_sentences:
            return ' '.join(sentences)

        # Use BERT for better sentence embeddings
        from sentence_transformers import SentenceTransformer
        model = SentenceTransformer('all-MiniLM-L6-v2')
        sentence_embeddings = model.encode(sentences)

        # Calculate similarity matrix
        similarity_matrix = cosine_similarity(sentence_embeddings)

        # Apply TextRank with damping (better for coherence)
        graph = nx.from_numpy_array(similarity_matrix)
        scores = nx.pagerank(graph, alpha=0.85)

        # Get top sentences and reorder
        ranked = sorted(((scores[i], i) for i in range(len(sentences))), reverse=True)
        top_indices = sorted([i for (score, i) in ranked[:num_sentences]])

        # Simple fusion of adjacent sentences
        summary = []
        for i in top_indices:
            if summary and i == top_indices[top_indices.index(i) - 1] + 1:
                summary[-1] += " " + sentences[i]
            else:
                summary.append(sentences[i])

        return ' '.join(summary)

    def abstractive_summarize(self, text, max_length=200, min_length=50):
        """Improved abstractive summary with better model"""
        # Use a fine-tuned model for better results
        model_name = "facebook/bart-large-cnn"
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModelForSeq2SeqLM.from_pretrained(model_name).to(self.device)

        inputs = tokenizer([text], max_length=1024, truncation=True, return_tensors="pt").to(self.device)
        summary_ids = model.generate(
            inputs["input_ids"],
            max_length=max_length,
            min_length=min_length,
            length_penalty=2.0,
            num_beams=4,
            early_stopping=True,
            no_repeat_ngram_size=3  # Avoid repetition
        )
        return tokenizer.decode(summary_ids[0], skip_special_tokens=True)

    def simplify_text(self, text):
        """Text simplification using T5"""
        input_text = "simplify: " + text
        input_ids = self.tokenizer.encode(input_text, return_tensors='pt', max_length=512, truncation=True).to(
            self.device)
        simplified_ids = self.model.generate(
            input_ids,
            max_length=len(input_ids[0]) + 50,
            num_beams=4,
            early_stopping=True
        )
        return self.tokenizer.decode(simplified_ids[0], skip_special_tokens=True)


class EvaluationMetrics:
    """Handles evaluation of summaries"""

    def __init__(self):
        self.scorer = rouge_scorer.RougeScorer(['rougeL'], use_stemmer=True)

    def calculate_rouge(self, reference, candidate):
        """Calculate ROUGE-L score"""
        return self.scorer.score(reference, candidate)['rougeL'].fmeasure

    def calculate_readability(self, text):
        """Calculate Flesch-Kincaid grade level"""
        return textstat.flesch_kincaid_grade(text)

    def evaluate_summary(self, original_text, summary_text, reference_summary=None):
        """Comprehensive evaluation of summary quality"""
        results = {}
        # Compression ratio
        original_length = len(word_tokenize(original_text))
        summary_length = len(word_tokenize(summary_text))
        results['compression_ratio'] = summary_length / original_length
        # Readability
        original_readability = self.calculate_readability(original_text)
        summary_readability = self.calculate_readability(summary_text)
        results['readability_improvement'] = original_readability - summary_readability
        # ROUGE score if reference available
        if reference_summary:
            results['rougeL'] = self.calculate_rouge(reference_summary, summary_text)
        return results


# Initialize components
cleaner = TextCleaner()
summarizer = Summarizer()
evaluator = EvaluationMetrics()


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/summarize', methods=['POST'])
def summarize():
    data = request.form
    text = data.get('text')
    file = request.files.get('file')
    url = data.get('url')
    num_sentences = int(data.get('num_sentences', 5))
    max_length = int(data.get('max_length', 200))
    min_length = int(data.get('min_length', 50))

    if not text and not file and not url:
        return jsonify({'error': 'No input provided'}), 400

    if file:
        if file.filename.lower().endswith('.pdf'):
            try:
                reader = PdfReader(io.BytesIO(file.read()))
                text = ""
                for page in reader.pages:
                    text += page.extract_text()
            except Exception as e:
                return jsonify({'error': f'Error reading PDF: {str(e)}'}), 400
        elif file.filename.lower().endswith('.txt'):
            try:
                text = file.read().decode('utf-8')
            except Exception as e:
                return jsonify({'error': f'Error reading text file: {str(e)}'}), 400
        else:
            return jsonify({'error': 'Unsupported file format'}), 400

    if url and not text:
        try:
            response = requests.get(url)
            response.raise_for_status()
            text = response.text
        except Exception as e:
            return jsonify({'error': f'Error fetching URL: {str(e)}'}), 400

    if not text.strip():
        return jsonify({'error': 'No text content found'}), 400

    # Clean the text
    cleaned_text = cleaner.clean_text(text)

    # Perform summarization
    extractive_summary = summarizer.extractive_summarize(cleaned_text, num_sentences)
    abstractive_summary = summarizer.abstractive_summarize(cleaned_text, max_length, min_length)
    simplified_text = summarizer.simplify_text(cleaned_text)

    # Evaluate the summaries
    evaluation_results = evaluator.evaluate_summary(cleaned_text, abstractive_summary)
    original_readability = evaluator.calculate_readability(cleaned_text)
    simplified_readability = evaluator.calculate_readability(simplified_text)

    return jsonify({
        'original_text': cleaned_text,
        'extractive_summary': extractive_summary,
        'abstractive_summary': abstractive_summary,
        'simplified_text': simplified_text,
        'evaluation': {
            'compression_ratio': evaluation_results['compression_ratio'],
            'readability_improvement': evaluation_results['readability_improvement'],
            'original_readability': original_readability,
            'simplified_readability': simplified_readability
        }
    })


if __name__ == '__main__':
    app.run(port=5001)
