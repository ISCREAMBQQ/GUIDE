import json
import os
import re
import string
from collections import Counter
from typing import List, Set, Tuple, Dict

# --- FIX 1: Set Matplotlib backend ---
# This line MUST come before importing pyplot
import matplotlib

matplotlib.use('Agg')  # Use a non-interactive backend suitable for servers
import matplotlib.pyplot as plt
# --- End of FIX 1 ---

import nltk
import spacy
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
from wordcloud import WordCloud


def create_and_save_wordcloud(word_counts: Dict[str, int], location_name: str, output_path: str):
    """
    Generates and saves a word cloud image from word frequency data.
    """
    if not word_counts:
        print(f"  - No words to generate a cloud for '{location_name}'. Skipping.")
        return

    wc = WordCloud(
        background_color='white',
        width=800,
        height=600,
        max_words=100,
        colormap='viridis',
        contour_width=1,
        contour_color='steelblue',
        random_state=42
    )
    wc.generate_from_frequencies(word_counts)

    plt.figure(figsize=(10, 7))
    plt.imshow(wc, interpolation='bilinear')
    plt.title(f'Keyword Cloud for: {location_name}', fontsize=16)
    plt.axis('off')
    plt.savefig(output_path, bbox_inches='tight')
    plt.close()


class KeywordExtractor:
    """
    A class dedicated to extracting keywords from text documents.
    """

    def __init__(self, stopwords_file_path: str):
        print("Initializing KeywordExtractor...")
        self._download_nltk_data()
        self.stop_words = self._load_stop_words(stopwords_file_path)
        self.lemmatizer = WordNetLemmatizer()
        print("KeywordExtractor ready.")

    def _download_nltk_data(self):
        required_packages = ['punkt', 'averaged_perceptron_tagger', 'wordnet', 'stopwords']
        for package in required_packages:
            try:
                nltk.data.find(f'tokenizers/{package}')
            except LookupError:
                print(f"Downloading NLTK package: {package}...")
                nltk.download(package, quiet=True)

    def _load_stop_words(self, file_path: str) -> Set[str]:
        stop_words = set(stopwords.words('english'))
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                custom_words = {line.strip().lower() for line in f if line.strip()}
            stop_words.update(custom_words)
        except FileNotFoundError:
            print(f"Warning: Custom stop words file '{file_path}' not found. Using NLTK defaults.")
        return stop_words

    def extract(self, reviews: List[str], category: str, num_keywords: int = 10) -> Tuple[List[str], Dict[str, int]]:
        full_text = ' '.join(reviews).lower()
        tokens = word_tokenize(full_text)
        punct = set(string.punctuation)
        filtered_tokens = [w for w in tokens if w.isalpha() and w not in self.stop_words and w not in punct]
        pos_tags = nltk.pos_tag(filtered_tokens)
        target_pos = {'JJ', 'JJR', 'JJS', 'NN', 'NNS'}
        descriptive_words = [word for word, tag in pos_tags if tag in target_pos]
        lemmatized_words = [self.lemmatizer.lemmatize(word) for word in descriptive_words]

        word_counts = Counter(lemmatized_words)
        word_counts_for_keywords = word_counts.copy()
        if category.lower() in word_counts_for_keywords:
            del word_counts_for_keywords[category.lower()]

        top_words = [word for word, count in word_counts_for_keywords.most_common(num_keywords - 1)]
        final_keywords = [category.capitalize()] + top_words

        return final_keywords, dict(word_counts)


class SemanticSimilarityCalculator:
    """
    A class dedicated to calculating semantic similarity between texts.
    """

    def __init__(self, stopwords_file_path: str):
        print("\nInitializing SemanticSimilarityCalculator...")
        self.nlp = self._load_spacy_model()
        self.stop_words = self._load_stop_words(stopwords_file_path)  # Reusing load logic
        print("SemanticSimilarityCalculator ready.")

    def _load_stop_words(self, file_path: str) -> Set[str]:
        """Loads default and custom stop words from a file."""
        # This logic is duplicated from KeywordExtractor to make this class standalone
        stop_words = set(stopwords.words('english'))
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                custom_words = {line.strip().lower() for line in f if line.strip()}
            stop_words.update(custom_words)
        except FileNotFoundError:
            print(f"Warning: Custom stop words file '{file_path}' not found. Using NLTK defaults.")
        return stop_words

    def _load_spacy_model(self):
        """
        Loads the spaCy model from a local path, with improved robustness.
        """
        # --- FIX 2: More robust path handling ---
        try:
            # This works when running as a script
            script_dir = os.path.dirname(os.path.abspath(__file__))
        except NameError:
            # This is a fallback for interactive environments (like notebooks)
            # where __file__ is not defined. It assumes the script is run from project root.
            script_dir = os.getcwd()
            # --- End of FIX 2 ---

        model_path = os.path.join(script_dir, 'models', 'en_core_web_md-3.7.1', 'en_core_web_md',
                                  'en_core_web_md-3.7.1')
        print(f"Attempting to load spaCy model from: {model_path}")

        if not os.path.exists(model_path):
            raise FileNotFoundError(
                f"SpaCy model folder not found at the expected path: {model_path}\n"
                f"Please ensure the model is downloaded and placed in the correct project directory."
            )
        return spacy.load(model_path)

    def calculate(self, text1: str, text2: str) -> float:
        if not text1.strip() or not text2.strip():
            return 0.0
        doc1 = self.nlp(text1)
        doc2 = self.nlp(text2)
        tokens1 = [t.text for t in doc1 if t.text.lower() not in self.stop_words and not t.is_punct]
        tokens2 = [t.text for t in doc2 if t.text.lower() not in self.stop_words and not t.is_punct]
        doc1_no_stops = self.nlp(' '.join(tokens1))
        doc2_no_stops = self.nlp(' '.join(tokens2))

        if not doc1_no_stops.vector.any() or not doc2_no_stops.vector.any():
            return 0.0

        return doc1_no_stops.similarity(doc2_no_stops)


if __name__ == '__main__':
    INPUT_JSON_PATH = 'Graph/GUIDE_037_updated.json'
    OUTPUT_JSON_PATH = 'Graph/GUIDE_0371_with_features.json'
    STOPWORDS_FILE_PATH = 'ENGLISH_STOP.txt'
    WORDCLOUD_DIR = 'wordclouds'

    os.makedirs(WORDCLOUD_DIR, exist_ok=True)
    print(f"Word cloud images will be saved in '{WORDCLOUD_DIR}/'")

    keyword_extractor = KeywordExtractor(stopwords_file_path=STOPWORDS_FILE_PATH)

    try:
        with open(INPUT_JSON_PATH, 'r', encoding='utf-8') as f:
            locations_data = json.load(f)
    except FileNotFoundError:
        print(f"FATAL ERROR: Input JSON file not found at '{INPUT_JSON_PATH}'. Please check the path.")
        exit()  # Exit the script if the main data file is missing

    print("\nProcessing locations, extracting features, and generating word clouds...")
    for location in locations_data:
        if 'reviews' in location and 'category' in location and location['reviews']:
            reviews = location['reviews']
            category = location['category']
            location_name = location['name']

            features, word_counts = keyword_extractor.extract(reviews, category, num_keywords=10)

            features.insert(0, location_name)
            location['feature'] = features

            clean_filename = re.sub(r'[\\/*?:"<>|]', "", location_name).replace(" ", "_")
            image_filename = f"{clean_filename}.png"
            image_path = os.path.join(WORDCLOUD_DIR, image_filename)

            create_and_save_wordcloud(word_counts, location_name, image_path)

            location['word_cloud'] = image_path

            print(f"  - Processed '{location_name}': Features extracted, word cloud saved to '{image_path}'")
        else:
            print(f"  - Skipping '{location.get('name', 'N/A')}' due to missing reviews or category.")

    with open(OUTPUT_JSON_PATH, 'w', encoding='utf-8') as f:
        json.dump(locations_data, f, indent=4)

    print(f"\nProcessing complete. Output saved to: {OUTPUT_JSON_PATH}")