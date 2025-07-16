import spacy
from collections import Counter
import string
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from typing import Tuple, Dict, List, Set


class KeywordExtractor:
    """
    A class dedicated to extracting keywords from text documents.
    It manages NLTK data and a custom stop words list.
    """

    def __init__(self, stopwords_file_path: str):
        """Initializes the extractor by loading necessary NLTK data and stop words."""
        print("Initializing KeywordExtractor...")
        self._download_nltk_data()
        self.stop_words = self._load_stop_words(stopwords_file_path)
        print("KeywordExtractor ready.")

    def _download_nltk_data(self):
        """Downloads required NLTK packages if not already present."""
        required_packages = ['punkt', 'averaged_perceptron_tagger', 'wordnet', 'stopwords']
        for package in required_packages:
            try:
                nltk.data.find(f'tokenizers/{package}')
            except LookupError:
                print(f"Downloading NLTK package: {package}...")
                nltk.download(package, quiet=True)

    def _load_stop_words(self, file_path: str) -> Set[str]:
        """Loads default and custom stop words from a file."""
        stop_words = set(stopwords.words('english'))
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                custom_words = {line.strip().lower() for line in f if line.strip()}
            stop_words.update(custom_words)
        except FileNotFoundError:
            print(f"Warning: Custom stop words file '{file_path}' not found. Using NLTK defaults.")
        return stop_words

    def extract(self, reviews: List[str], category: str, num_keywords: int = 10) -> List[str]:
        """
        Public interface to extract descriptive keywords from a list of reviews.
        """
        full_text = ' '.join(reviews).lower()
        from nltk.stem import WordNetLemmatizer
        tokens = word_tokenize(full_text)
        punct = set(string.punctuation)
        filtered_tokens = [w for w in tokens if w.isalpha() and w not in self.stop_words and w not in punct]
        pos_tags = nltk.pos_tag(filtered_tokens)
        target_pos = {'JJ', 'JJR', 'JJS', 'NN', 'NNS'}
        descriptive_words = [word for word, tag in pos_tags if tag in target_pos]
        lemmatizer = WordNetLemmatizer()
        lemmatized_words = [lemmatizer.lemmatize(word) for word in descriptive_words]
        word_counts = Counter(lemmatized_words)
        if category.lower() in word_counts:
            del word_counts[category.lower()]
        top_words = [word for word, count in word_counts.most_common(num_keywords - 1)]
        return [category.capitalize()] + top_words


class SemanticSimilarityCalculator:
    """
    A class dedicated to calculating semantic similarity between texts.
    It manages the spaCy model and a stop words list for consistent processing.
    """

    def __init__(self, stopwords_file_path: str):
        """Initializes the calculator by loading the spaCy model and stop words."""
        print("\nInitializing SemanticSimilarityCalculator...")
        self.nlp = self._load_spacy_model()
        # It also needs stop words to properly clean text before vector comparison
        self.stop_words = KeywordExtractor(stopwords_file_path).stop_words
        print("SemanticSimilarityCalculator ready.")

    def _load_spacy_model(self):
        """Loads the spaCy model, downloading if necessary."""
        try:
            return spacy.load("en_core_web_md")
        except OSError:
            print("Downloading spaCy model 'en_core_web_md'... (This may take a minute)")
            spacy.cli.download("en_core_web_md")
            return spacy.load("en_core_web_md")

    def calculate(self, text1: str, text2: str) -> float:
        """
        Public interface to calculate the semantic similarity between two texts.
        """
        if not text1.strip() or not text2.strip():
            return 0.0
        doc1 = self.nlp(text1)
        doc2 = self.nlp(text2)
        tokens1 = [t.text for t in doc1 if t.text.lower() not in self.stop_words and not t.is_punct]
        tokens2 = [t.text for t in doc2 if t.text.lower() not in self.stop_words and not t.is_punct]
        doc1_no_stops = self.nlp(' '.join(tokens1))
        doc2_no_stops = self.nlp(' '.join(tokens2))
        if not doc1_no_stops.has_vector or doc1_no_stops.vector_norm == 0 or \
                not doc2_no_stops.has_vector or doc2_no_stops.vector_norm == 0:
            return 0.0
        return doc1_no_stops.similarity(doc2_no_stops)

    # --- Main Execution Example ---


if __name__ == '__main__':
    # 1. Define the input data
    reviews_by_location = {
        "Sunny Bay Beach": ["beautiful sandy beach", "crystal clear water", "stunning sunset view"],
        "Green Meadows Park": ["quiet peaceful park", "long walk", "huge green space", "relaxing afternoon"],
        "City History Museum": ["vast collection of historical artifacts", "educational", "impressive building"]
    }
    STOPWORDS_FILE_PATH = 'stop_word.txt'
    with open(STOPWORDS_FILE_PATH, 'w', encoding='utf-8') as f:
        f.write("want\nfind\nplace\ngo\nsee\nexperience\nfeel\nlike\nnew\nbuilding\nmodern")

        # 2. Instantiate the two main classes
    keyword_extractor = KeywordExtractor(stopwords_file_path=STOPWORDS_FILE_PATH)
    similarity_calculator = SemanticSimilarityCalculator(stopwords_file_path=STOPWORDS_FILE_PATH)

    # 3. Use the KeywordExtractor to process all locations first
    print("\n--- Step 1: Extracting Keywords for all locations ---")
    location_concepts = {}
    for name, reviews in reviews_by_location.items():
        keywords = keyword_extractor.extract(reviews, name)
        location_concepts[name] = " ".join(keywords)
        print(f" > {name} Concept: {location_concepts[name]}")

        # 4. Use the SimilarityCalculator to find the best match for a user's need
    print("\n--- Step 2: Calculating Similarity for a User's Need ---")
    user_needs = "I feel like learning something new about the past."
    print(f"User Need: \"{user_needs}\"")

    scores = {}
    for name, concept in location_concepts.items():
        scores[name] = similarity_calculator.calculate(user_needs, concept)
        print(f" > Similarity with '{name}': {scores[name]:.2%}")

        # 5. Determine and print the final recommendation
    best_match = max(scores, key=scores.get)
    print(f"\nFinal Recommendation: The best match is '{best_match}' (Score: {scores[best_match]:.2%})")
