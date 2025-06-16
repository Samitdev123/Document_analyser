import os
import io
from flask import Flask, request, jsonify
from flask_cors import CORS
from docx import Document
import spacy
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import LatentDirichletAllocation
from transformers import pipeline
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)
CORS(app)

# Global variables for models (load once)
nlp = None
summarizer = None


def initialize_models():
    """Initialize NLP models on startup"""
    global nlp, summarizer
    try:
        logger.info("Loading spaCy model...")
        nlp = spacy.load("en_core_web_sm")
        logger.info("Loading summarization model...")
        summarizer = pipeline("summarization", model="facebook/bart-large-cnn")
        logger.info("Models loaded successfully!")
    except Exception as e:
        logger.error(f"Error loading models: {e}")
        raise e


def preprocess(text):
    """Preprocess text for LDA"""
    if nlp is None:
        raise ValueError("spaCy model not loaded")

    doc = nlp(text)
    return " ".join([token.lemma_.lower() for token in doc if token.is_alpha and not token.is_stop])

def summarize_long_text(text, chunk_size=2048):
    """Split long text into chunks and summarize each separately"""
    chunks = [text[i:i + chunk_size] for i in range(0, len(text), chunk_size)]
    summaries = []
    for chunk in chunks:
        if len(chunk.split()) < 10:
            continue
        result = summarizer(
            chunk,
            max_length=130,
            min_length=30,
            do_sample=False
        )
        summaries.append(result[0]['summary_text'])
    return " ".join(summaries)


def process_documents(files):
    """Process uploaded documents and return analysis results"""
    documents = []
    filenames = []

    # Extract text from uploaded files
    for file in files:
        try:
            doc = Document(io.BytesIO(file.read()))
            text = "\n".join([para.text.strip() for para in doc.paragraphs if para.text.strip()])
            if text:
                documents.append(text)
                filenames.append(file.filename)
        except Exception as e:
            logger.error(f"Error processing file {file.filename}: {e}")
            continue

    if not documents:
        return {"error": "No valid documents found"}

    logger.info(f"Processing {len(documents)} documents")

    # Preprocess documents
    processed_docs = [preprocess(doc) for doc in documents]

    # TF-IDF and LDA
    n_docs = len(processed_docs)

    # Avoid max_df being less than min_df when number of docs is small
    min_df_val = 1
    max_df_val = 0.8 if n_docs > 1 else 1.0  # allow all terms if only 1 doc

    vectorizer = TfidfVectorizer(max_features=1000, min_df=min_df_val, max_df=max_df_val)

    X = vectorizer.fit_transform(processed_docs)

    # Adjust number of topics based on document count
    n_topics = min(len(documents), 10)  # Max 10 topics
    lda = LatentDirichletAllocation(n_components=n_topics, random_state=42, max_iter=10)
    lda.fit(X)

    terms = vectorizer.get_feature_names_out()
    topic_matrix = lda.transform(X)

    results = {}

    # Generate summaries and extract topics
    for i, original_text in enumerate(documents):
        try:
            # Truncate text for summarization (BART has token limits)
            if len(original_text.split()) < 10:
                summary = "Document too short for meaningful summarization."
            else:
                summary = summarize_long_text(original_text)

            # Get dominant topic for this document
            topic_idx = topic_matrix[i].argmax()
            topic_keywords = ", ".join([
                terms[index] for index in lda.components_[topic_idx].argsort()[:-6:-1]
            ])

            results[topic_keywords] = summary

        except Exception as e:
            logger.error(f"Error processing document {i}: {e}")
            results[f"Document {i + 1} topics"] = "Error generating summary"

    return results

@app.route('/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    return jsonify({"status": "healthy", "models_loaded": nlp is not None and summarizer is not None})


@app.route('/analyze', methods=['POST'])
def analyze_documents():
    try:
        if 'documents' not in request.files:
            return jsonify({"error": "No documents provided"}), 400

        files = request.files.getlist('documents')

        if not files or all(file.filename == '' for file in files):
            return jsonify({"error": "No files selected"}), 400

        valid_files = [file for file in files if file.filename.endswith('.docx')]

        if not valid_files:
            return jsonify({"error": "No valid .docx files found"}), 400

        results = process_documents(valid_files)

        if isinstance(results, dict) and "error" in results:
            return jsonify(results), 400

        return jsonify({
            "success": True,
            "processed_count": len(valid_files),
            "documents": results
        })

    except Exception as e:
        logger.error(f"Error in analyze_documents: {e}")
        return jsonify({"error": f"Internal server error: {str(e)}"}), 500


if __name__ == '__main__':
    try:
        initialize_models()
        app.run(debug=False, host='0.0.0.0', port=5000)
    except Exception as e:
        logger.error(f"Failed to start server: {e}")
        print("Please ensure you have installed the required dependencies:")
        print("pip install flask flask-cors python-docx spacy scikit-learn transformers torch")
        print("python -m spacy download en_core_web_sm")