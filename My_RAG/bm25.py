from rank_bm25 import BM25Okapi
import jieba
import nltk
from nltk.stem import WordNetLemmatizer
#nltk.download('wordnet')
nltk.data.path.append('./nltk_data/')

import os

class BM25Retriever:
    def __init__(self, chunks, language="en"):
        self.chunks = chunks
        self.language = language
        self.corpus = [chunk['page_content'] for chunk in chunks]
        current_dir = os.path.dirname(os.path.abspath(__file__))
        self.stopwords = open(os.path.join(current_dir, 'EnglishStopwords.txt'), 'r').read().split()
        self.ch_stopwords = open(os.path.join(current_dir, 'ChineseStopwords.txt'), 'r', encoding="utf8").read().split()
        self.lemmatizer = WordNetLemmatizer()
        if language == "zh":
            #self.tokenized_corpus = [list(jieba.cut(doc)) for doc in self.corpus]
            self.tokenized_corpus = [self.get_weighted_tokens(doc, "zh") for doc in self.corpus]
        else:
            self.tokenized_corpus = [self.get_weighted_tokens(doc, "en") for doc in self.corpus]
        #print(self.tokenized_corpus)
        self.bm25 = BM25Okapi(self.tokenized_corpus)

    def retrieve(self, query, top_k=5):
        if self.language == "zh":
            #tokenized_query = list(jieba.cut(query))
            tokenized_query = self.get_weighted_tokens(query, "zh")
        else:
            #tokenized_query = query.lower().split(" ")
            tokenized_query = self.get_weighted_tokens(query, "en")
        top_chunks = self.bm25.get_top_n(tokenized_query, self.chunks, n=top_k)
        return top_chunks
        
    def retrieve_with_scores(self, query, top_k=5):
        if self.language == "zh":
            # tokenized_query = list(jieba.cut(query))
            tokenized_query = self.get_weighted_tokens(query, "zh")
        else:
            # tokenized_query = query.lower().split(" ")
            tokenized_query = self.get_weighted_tokens(query, "en")
        
        scores = self.bm25.get_scores(tokenized_query)
        # Get top_k indices
        top_n_indices = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:top_k]
        
        return [(self.chunks[i], scores[i]) for i in top_n_indices]


    def get_weighted_tokens(self, text, language):
        if language == "en":
            text = text.replace(".", "")
            text = text.replace(r"\s+", " ")
            tokens = text.lower().split()
        elif language == "zh":
            tokens = jieba.lcut(text)

        if language == "en":
            clean_tokens = [
                self.lemmatizer.lemmatize(token)
                for token in tokens
                if token not in self.stopwords and token.isalpha()
            ]
        elif language == "zh":
            clean_tokens = [token for token in tokens if token not in self.ch_stopwords]
        final_tokens = list(clean_tokens)

        #N-grams (Bigrams)
        if len(clean_tokens) > 1:
            bigrams = [" ".join(pair) for pair in zip(clean_tokens, clean_tokens[1:])]
            final_tokens.extend(bigrams)

        return final_tokens

    def create_retriever(chunks, language):
        """Creates a BM25 retriever from document chunks."""
        return BM25Retriever(chunks, language)
