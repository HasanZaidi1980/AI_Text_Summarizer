# AI_Text_Summarizer
This project contains a Flask web application that cleans text from various sources (direct input, PDF files, or URLs) and generates multiple types of summaries—extractive, abstractive, and simplified versions—using NLP models like BART and T5, while also evaluating summary quality through readability metrics and compression ratios.

## Features
- Multi-format input support (text, PDF, URL)
- Extractive summarization using BERT embeddings and TextRank
- Abstractive summarization using BART-Large-CNN
- Text simplification with T5
- Summary evaluation (ROUGE scores, readability metrics, compression ratios)
- Clean and responsive web interface

## Technologies
Flask, NLTK, scikit-learn, Transformers (Hugging Face), PyTorch, Sentence-Transformers, ROUGE, TextStat
