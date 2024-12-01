from flask import Flask, render_template, request, jsonify
import networkx as nx
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.corpus import stopwords
import nltk
nltk.download('punkt')
nltk.download('punkt_tab')


app = Flask(__name__)

# Define function to preprocess sentences
def preprocess_sentences(text):
    sentences = sent_tokenize(text)
    stop_words = set(stopwords.words('english'))
    processed = [[word.lower() for word in word_tokenize(sentence) if word.isalnum() and word.lower() not in stop_words]
                 for sentence in sentences]
    return sentences, processed

# Build similarity matrix
def build_similarity_matrix(sentences):
    # Create a TF-IDF vectorizer
    vectorizer = TfidfVectorizer()

    # Fit and transform the sentences to get their TF-IDF vectors
    tfidf_matrix = vectorizer.fit_transform([' '.join(sentence) for sentence in sentences]) # Join the tokens back into sentences

    # Calculate cosine similarity using the TF-IDF vectors
    similarity_matrix = cosine_similarity(tfidf_matrix, tfidf_matrix)
    return similarity_matrix

# Summarization with TextRank
def summarize_text(text, top_n=5):
    original_sentences, processed_sentences = preprocess_sentences(text)
    similarity_matrix = build_similarity_matrix(processed_sentences)
    nx_graph = nx.from_numpy_array(similarity_matrix)
    scores = nx.pagerank(nx_graph)
    ranked_sentences = sorted(((scores[i], s) for i, s in enumerate(original_sentences)), reverse=True)
    summary = " ".join([sentence for _, sentence in ranked_sentences[:top_n]])
    return summary

@app.route('/')
def index():
    print("Root route accessed")
    return render_template('index.html')

@app.route('/summarize', methods=['POST'])
def summarize():
    text = request.form['text']
    print("Received Text:", text)  # Debugging line
    try:
        summary = summarize_text(text)
        print("Generated Summary:", summary)  # Debugging line
        return jsonify({'summary': summary})
    except Exception as e:
        print("Error:", e)  # Debugging line
        return jsonify({'error': str(e)}), 500
    

if __name__ == "__main__":
    app.run(debug=True)
