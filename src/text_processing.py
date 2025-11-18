"""
텍스트 처리 모듈

자연어 텍스트를 신경망이 처리할 수 있는 형태로 변환합니다.
"""

import numpy as np
import re
from collections import Counter


class Tokenizer:
    """
    텍스트를 토큰으로 분리하는 토크나이저
    """

    def __init__(self, max_words=1000, lowercase=True):
        """
        Args:
            max_words: 최대 단어 수
            lowercase: 소문자 변환 여부
        """
        self.max_words = max_words
        self.lowercase = lowercase
        self.word_index = {}
        self.index_word = {}
        self.word_counts = Counter()

    def fit(self, texts):
        """
        텍스트로부터 단어 사전 구축

        Args:
            texts: 텍스트 리스트
        """
        for text in texts:
            tokens = self.tokenize(text)
            self.word_counts.update(tokens)

        # 빈도 높은 단어부터 인덱스 할당
        # 0: 패딩, 1: 미등록 단어
        self.word_index = {'<PAD>': 0, '<UNK>': 1}
        self.index_word = {0: '<PAD>', 1: '<UNK>'}

        for idx, (word, _) in enumerate(
            self.word_counts.most_common(self.max_words - 2), 2
        ):
            self.word_index[word] = idx
            self.index_word[idx] = word

    def tokenize(self, text):
        """
        텍스트를 토큰으로 분리

        Args:
            text: 입력 텍스트

        Returns:
            토큰 리스트
        """
        if self.lowercase:
            text = text.lower()

        # 한글, 영문, 숫자만 유지
        text = re.sub(r'[^가-힣a-zA-Z0-9\s]', '', text)

        # 공백으로 분리
        tokens = text.split()

        return tokens

    def texts_to_sequences(self, texts):
        """
        텍스트를 숫자 시퀀스로 변환

        Args:
            texts: 텍스트 리스트

        Returns:
            숫자 시퀀스 리스트
        """
        sequences = []
        for text in texts:
            tokens = self.tokenize(text)
            sequence = [
                self.word_index.get(token, 1)  # 미등록 단어는 1
                for token in tokens
            ]
            sequences.append(sequence)

        return sequences

    def sequences_to_texts(self, sequences):
        """
        숫자 시퀀스를 텍스트로 변환

        Args:
            sequences: 숫자 시퀀스 리스트

        Returns:
            텍스트 리스트
        """
        texts = []
        for sequence in sequences:
            tokens = [
                self.index_word.get(idx, '<UNK>')
                for idx in sequence
                if idx != 0  # 패딩 제외
            ]
            texts.append(' '.join(tokens))

        return texts


class BagOfWords:
    """
    Bag of Words 벡터화
    """

    def __init__(self, tokenizer):
        """
        Args:
            tokenizer: Tokenizer 인스턴스
        """
        self.tokenizer = tokenizer
        self.vocab_size = len(tokenizer.word_index)

    def transform(self, texts):
        """
        텍스트를 BoW 벡터로 변환

        Args:
            texts: 텍스트 리스트

        Returns:
            BoW 벡터 행렬 (n_samples, vocab_size)
        """
        sequences = self.tokenizer.texts_to_sequences(texts)
        vectors = np.zeros((len(sequences), self.vocab_size))

        for i, sequence in enumerate(sequences):
            for word_idx in sequence:
                if word_idx < self.vocab_size:
                    vectors[i, word_idx] += 1

        return vectors


class TfidfVectorizer:
    """
    TF-IDF 벡터화

    TF-IDF = Term Frequency * Inverse Document Frequency
    """

    def __init__(self, tokenizer):
        """
        Args:
            tokenizer: Tokenizer 인스턴스
        """
        self.tokenizer = tokenizer
        self.vocab_size = len(tokenizer.word_index)
        self.idf = None

    def fit(self, texts):
        """
        IDF 계산

        Args:
            texts: 학습 텍스트 리스트
        """
        sequences = self.tokenizer.texts_to_sequences(texts)
        n_docs = len(sequences)

        # 각 단어가 등장하는 문서 수 계산
        doc_freq = np.zeros(self.vocab_size)
        for sequence in sequences:
            unique_words = set(sequence)
            for word_idx in unique_words:
                if word_idx < self.vocab_size:
                    doc_freq[word_idx] += 1

        # IDF 계산: log(전체 문서 수 / 단어가 등장하는 문서 수)
        self.idf = np.log((n_docs + 1) / (doc_freq + 1)) + 1

    def transform(self, texts):
        """
        텍스트를 TF-IDF 벡터로 변환

        Args:
            texts: 텍스트 리스트

        Returns:
            TF-IDF 벡터 행렬
        """
        sequences = self.tokenizer.texts_to_sequences(texts)
        vectors = np.zeros((len(sequences), self.vocab_size))

        for i, sequence in enumerate(sequences):
            # TF 계산
            tf = Counter(sequence)
            for word_idx, count in tf.items():
                if word_idx < self.vocab_size:
                    vectors[i, word_idx] = count

            # 정규화 (문서 길이로 나눔)
            if len(sequence) > 0:
                vectors[i] /= len(sequence)

        # IDF 곱하기
        if self.idf is not None:
            vectors *= self.idf

        return vectors


class WordEmbedding:
    """
    간단한 Word Embedding (학습 가능)

    단어를 밀집 벡터로 표현합니다.
    """

    def __init__(self, vocab_size, embedding_dim=50):
        """
        Args:
            vocab_size: 어휘 크기
            embedding_dim: 임베딩 차원
        """
        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim

        # 임베딩 행렬 초기화 (Xavier)
        limit = np.sqrt(6 / (vocab_size + embedding_dim))
        self.embeddings = np.random.uniform(
            -limit, limit,
            (vocab_size, embedding_dim)
        )

    def embed(self, sequences):
        """
        시퀀스를 임베딩 벡터로 변환

        Args:
            sequences: 단어 인덱스 시퀀스 리스트

        Returns:
            임베딩 벡터 행렬
        """
        embedded = []
        for sequence in sequences:
            # 시퀀스의 각 단어를 임베딩하고 평균
            if len(sequence) == 0:
                embedded.append(np.zeros(self.embedding_dim))
            else:
                word_vectors = [
                    self.embeddings[idx]
                    for idx in sequence
                    if idx < self.vocab_size
                ]
                if word_vectors:
                    embedded.append(np.mean(word_vectors, axis=0))
                else:
                    embedded.append(np.zeros(self.embedding_dim))

        return np.array(embedded)


def cosine_similarity(v1, v2):
    """
    코사인 유사도 계산

    Args:
        v1, v2: 벡터

    Returns:
        코사인 유사도 (-1 ~ 1)
    """
    norm1 = np.linalg.norm(v1)
    norm2 = np.linalg.norm(v2)

    if norm1 == 0 or norm2 == 0:
        return 0.0

    return np.dot(v1, v2) / (norm1 * norm2)


def extract_keywords(text, tokenizer, top_n=5):
    """
    텍스트에서 키워드 추출

    Args:
        text: 입력 텍스트
        tokenizer: Tokenizer 인스턴스
        top_n: 추출할 키워드 개수

    Returns:
        키워드 리스트
    """
    tokens = tokenizer.tokenize(text)
    word_freq = Counter(tokens)

    # 빈도 높은 단어 반환
    keywords = [word for word, _ in word_freq.most_common(top_n)]

    return keywords
