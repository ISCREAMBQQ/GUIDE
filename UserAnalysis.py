import json
import os

import spacy
from collections import Counter
import string
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from typing import List, Set


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
            # We want 9 keywords from frequency analysis, plus the category.
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
        """
        (核心修改)
        从项目中指定的本地文件夹路径加载spaCy模型。
        这种方法完全离线，非常可靠。
        """
        # 1. 获取当前文件(UserAnalysis.py)所在的目录
        current_dir = os.path.dirname(os.path.abspath(__file__))

        # 2. 构建模型的绝对路径 (相对于当前文件)
        #    这里的路径是 '..', 'models', 'en_core_web_sm-3.7.1'
        #    你需要根据你的实际文件夹名称进行调整
        model_path = os.path.join(current_dir, 'models', 'en_core_web_md-3.7.1', 'en_core_web_md',
                                  'en_core_web_md-3.7.1')
        # 如果你的models文件夹和UserAnalysis.py在同一级，路径应为：
        # model_path = os.path.join(current_dir, 'models', 'en_core_web_sm-3.7.1')

        print(f"Loading spaCy model from local path: {model_path}")

        # 3. 检查路径是否存在，提供清晰的错误提示
        if not os.path.exists(model_path):
            raise FileNotFoundError(
                f"SpaCy模型文件夹未在指定路径找到: {model_path}\n"
                f"请确认你已经下载并解压了模型，并将其放置在正确的项目目录中。"
            )

        # 4. 直接从该路径加载模型
        return spacy.load(model_path)

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
    INPUT_JSON_PATH = 'Graph/GUIDE038.json'
    OUTPUT_JSON_PATH = 'Graph/GUIDE038.json'
    STOPWORDS_FILE_PATH = 'ENGLISH_STOP.txt'

    # 4. Instantiate the extractor
    keyword_extractor = KeywordExtractor(stopwords_file_path=STOPWORDS_FILE_PATH)

    # 5. Load the JSON data
    with open(INPUT_JSON_PATH, 'r', encoding='utf-8') as f:
        locations_data = json.load(f)

    print("\nProcessing locations and extracting features...")
    # 6. Process each location to add the 'feature' list
    for location in locations_data:
        if 'reviews' in location and 'category' in location:
            reviews = location['reviews']
            category = location['category']
            # Extract 10 keywords (1 category + 9 from reviews)
            features = keyword_extractor.extract(reviews, category, num_keywords=10)
            features.insert(0, location['name'])
            location['feature'] = features
            print(f"  - Extracted features for '{location['name']}': {features}")

            # 7. Write the updated data to a new JSON file
    with open(OUTPUT_JSON_PATH, 'w', encoding='utf-8') as f:
        json.dump(locations_data, f, indent=4)

    print(f"\nProcessing complete. Output saved to: {OUTPUT_JSON_PATH}")
