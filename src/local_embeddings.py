"""
Local Embeddings Module - Offline Embedding Model Serving
로컬 임베딩 모델 서빙 (인터넷 연결 없이 사용 가능)

This module provides interface for running embedding models locally
using sentence-transformers. Perfect for semantic search, clustering,
and RAG (Retrieval Augmented Generation) applications.

Supported Models:
- all-MiniLM-L6-v2 (lightweight, fast)
- paraphrase-multilingual-MiniLM-L12-v2 (multilingual)
- all-mpnet-base-v2 (high quality)
- sentence-t5-base (large, best quality)

Usage:
    from src.local_embeddings import LocalEmbeddings

    # Initialize with model
    embedder = LocalEmbeddings(model_name="all-MiniLM-L6-v2")

    # Generate embeddings
    texts = ["Hello world", "Machine learning is great"]
    embeddings = embedder.encode(texts)

    # Calculate similarity
    similarity = embedder.cosine_similarity(embeddings[0], embeddings[1])
"""

import os
import numpy as np
from typing import List, Union, Optional, Dict
import warnings


class LocalEmbeddings:
    """
    로컬 임베딩 모델 인터페이스

    sentence-transformers를 사용하여 텍스트 임베딩을 생성합니다.
    한번 다운로드하면 인터넷 연결 없이 사용 가능합니다.
    """

    # 추천 모델 목록
    RECOMMENDED_MODELS = {
        "mini": "all-MiniLM-L6-v2",  # 가볍고 빠름 (80MB, 384 dim)
        "multilingual": "paraphrase-multilingual-MiniLM-L12-v2",  # 다국어 지원 (420MB, 384 dim)
        "base": "all-mpnet-base-v2",  # 높은 품질 (420MB, 768 dim)
        "large": "sentence-t5-base",  # 최고 품질 (220MB, 768 dim)
        "korean": "jhgan/ko-sroberta-multitask",  # 한국어 특화 (300MB, 768 dim)
    }

    def __init__(
        self,
        model_name: str = "all-MiniLM-L6-v2",
        device: str = "cpu",
        cache_folder: Optional[str] = None,
        verbose: bool = False
    ):
        """
        로컬 임베딩 모델 초기화

        Parameters:
        -----------
        model_name : str, default="all-MiniLM-L6-v2"
            사용할 모델 이름
            - 직접 모델 이름 지정 또는
            - 별칭 사용: "mini", "multilingual", "base", "large", "korean"

        device : str, default="cpu"
            실행 디바이스 ("cpu", "cuda", "mps")

        cache_folder : str, optional
            모델 캐시 폴더 경로 (기본: ~/.cache/torch/sentence_transformers)

        verbose : bool, default=False
            상세 로그 출력 여부
        """
        # 별칭을 실제 모델 이름으로 변환
        if model_name in self.RECOMMENDED_MODELS:
            actual_model_name = self.RECOMMENDED_MODELS[model_name]
            if verbose:
                print(f"별칭 '{model_name}' -> '{actual_model_name}'")
            model_name = actual_model_name

        self.model_name = model_name
        self.device = device
        self.cache_folder = cache_folder
        self.verbose = verbose
        self.model = None
        self.embedding_dim = None

        self._load_model()

    def _load_model(self):
        """모델 로드"""
        try:
            from sentence_transformers import SentenceTransformer
        except ImportError:
            raise ImportError(
                "sentence-transformers가 설치되지 않았습니다.\n"
                "설치: pip install sentence-transformers"
            )

        if self.verbose:
            print(f"임베딩 모델 로딩 중: {self.model_name}")
            print(f"- 디바이스: {self.device}")
            if self.cache_folder:
                print(f"- 캐시 폴더: {self.cache_folder}")

        try:
            self.model = SentenceTransformer(
                self.model_name,
                device=self.device,
                cache_folder=self.cache_folder
            )

            # Get embedding dimension
            self.embedding_dim = self.model.get_sentence_embedding_dimension()

            if self.verbose:
                print(f"✓ 모델 로딩 완료! (차원: {self.embedding_dim})")

        except Exception as e:
            raise RuntimeError(
                f"모델 로딩 실패: {e}\n\n"
                f"추천 모델:\n"
                f"  - mini: 가볍고 빠름\n"
                f"  - multilingual: 다국어 지원\n"
                f"  - korean: 한국어 특화\n"
                f"  - base: 높은 품질\n"
                f"  - large: 최고 품질"
            )

    def encode(
        self,
        texts: Union[str, List[str]],
        batch_size: int = 32,
        show_progress_bar: bool = False,
        normalize: bool = True
    ) -> np.ndarray:
        """
        텍스트를 임베딩 벡터로 변환

        Parameters:
        -----------
        texts : str or List[str]
            임베딩할 텍스트 (단일 문자열 또는 리스트)

        batch_size : int, default=32
            배치 크기

        show_progress_bar : bool, default=False
            진행률 표시 여부

        normalize : bool, default=True
            L2 정규화 수행 여부 (코사인 유사도 계산에 유리)

        Returns:
        --------
        np.ndarray
            임베딩 벡터
            - 단일 텍스트: shape (embedding_dim,)
            - 여러 텍스트: shape (n_texts, embedding_dim)
        """
        if self.model is None:
            raise RuntimeError("모델이 로드되지 않았습니다.")

        # Handle single string
        single_input = isinstance(texts, str)
        if single_input:
            texts = [texts]

        # Generate embeddings
        embeddings = self.model.encode(
            texts,
            batch_size=batch_size,
            show_progress_bar=show_progress_bar,
            normalize_embeddings=normalize,
            convert_to_numpy=True
        )

        # Return single vector for single input
        if single_input:
            return embeddings[0]

        return embeddings

    def encode_batch(
        self,
        texts: List[str],
        batch_size: int = 32,
        show_progress: bool = True
    ) -> np.ndarray:
        """
        대량의 텍스트를 배치로 임베딩 (진행률 표시)

        Parameters:
        -----------
        texts : List[str]
            임베딩할 텍스트 리스트

        batch_size : int, default=32
            배치 크기

        show_progress : bool, default=True
            진행률 표시 여부

        Returns:
        --------
        np.ndarray
            임베딩 행렬 (n_texts, embedding_dim)
        """
        return self.encode(
            texts,
            batch_size=batch_size,
            show_progress_bar=show_progress,
            normalize=True
        )

    @staticmethod
    def cosine_similarity(
        embedding1: np.ndarray,
        embedding2: np.ndarray
    ) -> float:
        """
        두 임베딩 간의 코사인 유사도 계산

        Parameters:
        -----------
        embedding1 : np.ndarray
            첫 번째 임베딩 벡터

        embedding2 : np.ndarray
            두 번째 임베딩 벡터

        Returns:
        --------
        float
            코사인 유사도 (-1.0 ~ 1.0)
            1.0 = 완전히 같음, 0.0 = 무관함, -1.0 = 정반대
        """
        # Normalize if needed
        norm1 = np.linalg.norm(embedding1)
        norm2 = np.linalg.norm(embedding2)

        if norm1 == 0 or norm2 == 0:
            return 0.0

        embedding1 = embedding1 / norm1
        embedding2 = embedding2 / norm2

        return float(np.dot(embedding1, embedding2))

    def semantic_search(
        self,
        query: str,
        corpus: List[str],
        top_k: int = 5,
        return_scores: bool = True
    ) -> List[Dict]:
        """
        쿼리와 가장 유사한 문서 검색

        Parameters:
        -----------
        query : str
            검색 쿼리

        corpus : List[str]
            검색 대상 문서 리스트

        top_k : int, default=5
            반환할 최대 결과 수

        return_scores : bool, default=True
            유사도 점수 포함 여부

        Returns:
        --------
        List[Dict]
            검색 결과
            [
                {"index": 0, "text": "...", "score": 0.95},
                {"index": 2, "text": "...", "score": 0.87},
                ...
            ]
        """
        # Encode query and corpus
        query_embedding = self.encode(query, normalize=True)
        corpus_embeddings = self.encode(corpus, normalize=True)

        # Calculate similarities
        similarities = np.dot(corpus_embeddings, query_embedding)

        # Get top-k indices
        top_indices = np.argsort(similarities)[::-1][:top_k]

        # Build results
        results = []
        for idx in top_indices:
            result = {
                "index": int(idx),
                "text": corpus[idx]
            }
            if return_scores:
                result["score"] = float(similarities[idx])
            results.append(result)

        return results

    def cluster(
        self,
        texts: List[str],
        n_clusters: int = 5,
        method: str = "kmeans"
    ) -> Dict:
        """
        텍스트 클러스터링

        Parameters:
        -----------
        texts : List[str]
            클러스터링할 텍스트 리스트

        n_clusters : int, default=5
            클러스터 개수

        method : str, default="kmeans"
            클러스터링 방법 ("kmeans" 또는 "agglomerative")

        Returns:
        --------
        Dict
            클러스터링 결과
            {
                "labels": [0, 1, 0, 2, ...],  # 각 텍스트의 클러스터 번호
                "clusters": {
                    0: ["text1", "text3", ...],
                    1: ["text2", ...],
                    ...
                }
            }
        """
        try:
            from sklearn.cluster import KMeans, AgglomerativeClustering
        except ImportError:
            raise ImportError(
                "scikit-learn이 필요합니다.\n"
                "설치: pip install scikit-learn"
            )

        # Generate embeddings
        embeddings = self.encode(texts, normalize=True)

        # Perform clustering
        if method == "kmeans":
            clusterer = KMeans(n_clusters=n_clusters, random_state=42)
        elif method == "agglomerative":
            clusterer = AgglomerativeClustering(n_clusters=n_clusters)
        else:
            raise ValueError(f"Unknown method: {method}")

        labels = clusterer.fit_predict(embeddings)

        # Organize results
        clusters = {}
        for idx, label in enumerate(labels):
            label = int(label)
            if label not in clusters:
                clusters[label] = []
            clusters[label].append(texts[idx])

        return {
            "labels": labels.tolist(),
            "clusters": clusters
        }

    def find_duplicates(
        self,
        texts: List[str],
        threshold: float = 0.95
    ) -> List[Dict]:
        """
        유사한 텍스트 찾기 (중복 검출)

        Parameters:
        -----------
        texts : List[str]
            검사할 텍스트 리스트

        threshold : float, default=0.95
            유사도 임계값 (0.0 ~ 1.0)

        Returns:
        --------
        List[Dict]
            중복 쌍 리스트
            [
                {"index1": 0, "index2": 3, "similarity": 0.97,
                 "text1": "...", "text2": "..."},
                ...
            ]
        """
        # Generate embeddings
        embeddings = self.encode(texts, normalize=True)

        # Calculate pairwise similarities
        similarities = np.dot(embeddings, embeddings.T)

        # Find duplicates
        duplicates = []
        n = len(texts)

        for i in range(n):
            for j in range(i + 1, n):
                sim = similarities[i, j]
                if sim >= threshold:
                    duplicates.append({
                        "index1": i,
                        "index2": j,
                        "similarity": float(sim),
                        "text1": texts[i],
                        "text2": texts[j]
                    })

        # Sort by similarity
        duplicates.sort(key=lambda x: x["similarity"], reverse=True)

        return duplicates

    def get_model_info(self) -> Dict:
        """
        모델 정보 반환

        Returns:
        --------
        Dict
            모델 메타데이터
        """
        return {
            "model_name": self.model_name,
            "embedding_dim": self.embedding_dim,
            "device": self.device,
            "cache_folder": self.cache_folder,
            "loaded": self.model is not None
        }

    def save_embeddings(
        self,
        embeddings: np.ndarray,
        filepath: str
    ):
        """
        임베딩을 파일로 저장

        Parameters:
        -----------
        embeddings : np.ndarray
            저장할 임베딩

        filepath : str
            저장 경로 (.npy 또는 .npz)
        """
        if filepath.endswith('.npz'):
            np.savez_compressed(filepath, embeddings=embeddings)
        else:
            np.save(filepath, embeddings)

        if self.verbose:
            print(f"✓ 임베딩 저장: {filepath}")

    @staticmethod
    def load_embeddings(filepath: str) -> np.ndarray:
        """
        파일에서 임베딩 로드

        Parameters:
        -----------
        filepath : str
            임베딩 파일 경로

        Returns:
        --------
        np.ndarray
            로드된 임베딩
        """
        if filepath.endswith('.npz'):
            data = np.load(filepath)
            return data['embeddings']
        else:
            return np.load(filepath)

    def __repr__(self):
        return f"LocalEmbeddings(model='{self.model_name}', dim={self.embedding_dim})"


class EmbeddingCache:
    """
    임베딩 캐시 - 동일한 텍스트를 다시 임베딩하지 않도록 캐싱
    """

    def __init__(
        self,
        embedder: LocalEmbeddings,
        max_cache_size: int = 10000
    ):
        """
        캐시 초기화

        Parameters:
        -----------
        embedder : LocalEmbeddings
            임베딩 모델

        max_cache_size : int, default=10000
            최대 캐시 크기
        """
        self.embedder = embedder
        self.max_cache_size = max_cache_size
        self.cache: Dict[str, np.ndarray] = {}

    def encode(self, text: str) -> np.ndarray:
        """
        캐시를 사용하여 임베딩 생성

        Parameters:
        -----------
        text : str
            임베딩할 텍스트

        Returns:
        --------
        np.ndarray
            임베딩 벡터
        """
        # Check cache
        if text in self.cache:
            return self.cache[text]

        # Generate embedding
        embedding = self.embedder.encode(text)

        # Update cache
        if len(self.cache) >= self.max_cache_size:
            # Remove oldest entry (FIFO)
            oldest_key = next(iter(self.cache))
            del self.cache[oldest_key]

        self.cache[text] = embedding

        return embedding

    def clear_cache(self):
        """캐시 초기화"""
        self.cache.clear()

    def get_cache_size(self) -> int:
        """캐시 크기 반환"""
        return len(self.cache)

    def __repr__(self):
        return f"EmbeddingCache(size={len(self.cache)}/{self.max_cache_size})"


# Utility functions
def compare_texts(
    text1: str,
    text2: str,
    model_name: str = "mini"
) -> float:
    """
    두 텍스트의 유사도를 빠르게 계산

    Parameters:
    -----------
    text1, text2 : str
        비교할 텍스트

    model_name : str, default="mini"
        사용할 모델 (별칭 또는 전체 이름)

    Returns:
    --------
    float
        유사도 점수 (0.0 ~ 1.0)
    """
    embedder = LocalEmbeddings(model_name=model_name)
    embeddings = embedder.encode([text1, text2])
    return LocalEmbeddings.cosine_similarity(embeddings[0], embeddings[1])


def list_available_models():
    """
    사용 가능한 추천 모델 목록 출력
    """
    print("\n추천 임베딩 모델:")
    print("=" * 70)

    models_info = [
        ("mini", "all-MiniLM-L6-v2", "80MB", "384", "가볍고 빠름, 일반 용도"),
        ("multilingual", "paraphrase-multilingual-MiniLM-L12-v2", "420MB", "384", "50+ 언어 지원"),
        ("korean", "jhgan/ko-sroberta-multitask", "300MB", "768", "한국어 특화"),
        ("base", "all-mpnet-base-v2", "420MB", "768", "높은 품질, 영어 최적화"),
        ("large", "sentence-t5-base", "220MB", "768", "최고 품질, 영어 최적화"),
    ]

    for alias, name, size, dim, desc in models_info:
        print(f"\n별칭: {alias}")
        print(f"  모델: {name}")
        print(f"  크기: {size} | 차원: {dim}")
        print(f"  설명: {desc}")

    print("\n" + "=" * 70)
    print("\n사용 예:")
    print('  embedder = LocalEmbeddings(model_name="mini")')
    print('  또는')
    print('  embedder = LocalEmbeddings(model_name="all-MiniLM-L6-v2")')
    print()


if __name__ == "__main__":
    # 사용 가능한 모델 목록
    list_available_models()

    # 사용 예제
    print("\n\n사용 예제:")
    print("-" * 70)
    print("""
    from src.local_embeddings import LocalEmbeddings

    # 1. 기본 임베딩 생성
    embedder = LocalEmbeddings(model_name="mini")
    texts = ["안녕하세요", "Hello world", "Machine learning"]
    embeddings = embedder.encode(texts)
    print(embeddings.shape)  # (3, 384)

    # 2. 유사도 계산
    similarity = embedder.cosine_similarity(embeddings[0], embeddings[1])
    print(f"유사도: {similarity:.3f}")

    # 3. 시맨틱 검색
    query = "AI and deep learning"
    corpus = [
        "Machine learning is a subset of AI",
        "I love pizza",
        "Neural networks are powerful",
        "The weather is nice today"
    ]
    results = embedder.semantic_search(query, corpus, top_k=2)
    for r in results:
        print(f"{r['text']} (score: {r['score']:.3f})")

    # 4. 중복 찾기
    documents = ["Text A", "Text A with minor change", "Different text"]
    duplicates = embedder.find_duplicates(documents, threshold=0.9)
    for dup in duplicates:
        print(f"Similar: {dup['text1']} <-> {dup['text2']} ({dup['similarity']:.3f})")
    """)
