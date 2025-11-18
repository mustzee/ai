"""
Local AI Server - REST API for Local Models
로컬 AI 서버 (인터넷 연결 없이 사용 가능)

Simple Flask-based REST API server for local LLM and embedding models.
All processing happens locally without any internet connection.

Endpoints:
    POST /generate - Text generation
    POST /chat - Chat completion
    POST /embed - Text embedding
    POST /search - Semantic search
    GET /models - List loaded models
    GET /health - Health check

Usage:
    python src/local_server.py --llm-model path/to/model.gguf --port 8000
"""

import argparse
import os
import sys
from typing import Dict, List, Optional
import json


def create_app(
    llm_model_path: Optional[str] = None,
    embedding_model: str = "mini",
    enable_llm: bool = True,
    enable_embeddings: bool = True,
    verbose: bool = False
):
    """
    Flask 앱 생성

    Parameters:
    -----------
    llm_model_path : str, optional
        LLM 모델 파일 경로

    embedding_model : str, default="mini"
        임베딩 모델 이름

    enable_llm : bool, default=True
        LLM 활성화 여부

    enable_embeddings : bool, default=True
        임베딩 활성화 여부

    verbose : bool, default=False
        상세 로그 출력

    Returns:
    --------
    Flask app
    """
    try:
        from flask import Flask, request, jsonify
    except ImportError:
        raise ImportError(
            "Flask가 설치되지 않았습니다.\n"
            "설치: pip install flask"
        )

    app = Flask(__name__)

    # Initialize models
    llm = None
    embedder = None

    if enable_llm and llm_model_path:
        try:
            from src.local_llm import LocalLLM
            print(f"LLM 로딩 중: {llm_model_path}")
            llm = LocalLLM(
                model_path=llm_model_path,
                n_ctx=2048,
                verbose=verbose
            )
            print("✓ LLM 로딩 완료!")
        except Exception as e:
            print(f"경고: LLM 로딩 실패: {e}")
            enable_llm = False

    if enable_embeddings:
        try:
            from src.local_embeddings import LocalEmbeddings
            print(f"임베딩 모델 로딩 중: {embedding_model}")
            embedder = LocalEmbeddings(
                model_name=embedding_model,
                verbose=verbose
            )
            print("✓ 임베딩 모델 로딩 완료!")
        except Exception as e:
            print(f"경고: 임베딩 모델 로딩 실패: {e}")
            enable_embeddings = False

    # Store corpus for semantic search
    corpus_store: Dict[str, List[str]] = {}

    # Routes
    @app.route("/health", methods=["GET"])
    def health():
        """헬스 체크"""
        return jsonify({
            "status": "healthy",
            "llm_enabled": enable_llm and llm is not None,
            "embeddings_enabled": enable_embeddings and embedder is not None
        })

    @app.route("/models", methods=["GET"])
    def list_models():
        """로드된 모델 정보"""
        models = {}

        if llm is not None:
            models["llm"] = llm.get_model_info()

        if embedder is not None:
            models["embeddings"] = embedder.get_model_info()

        return jsonify(models)

    @app.route("/generate", methods=["POST"])
    def generate():
        """텍스트 생성"""
        if not enable_llm or llm is None:
            return jsonify({"error": "LLM이 활성화되지 않았습니다"}), 400

        data = request.json
        prompt = data.get("prompt")

        if not prompt:
            return jsonify({"error": "prompt가 필요합니다"}), 400

        try:
            response = llm.generate(
                prompt=prompt,
                max_tokens=data.get("max_tokens", 256),
                temperature=data.get("temperature", 0.7),
                top_p=data.get("top_p", 0.9),
                top_k=data.get("top_k", 40),
                repeat_penalty=data.get("repeat_penalty", 1.1)
            )

            return jsonify({
                "response": response,
                "model": os.path.basename(llm.model_path)
            })

        except Exception as e:
            return jsonify({"error": str(e)}), 500

    @app.route("/chat", methods=["POST"])
    def chat():
        """채팅 완성"""
        if not enable_llm or llm is None:
            return jsonify({"error": "LLM이 활성화되지 않았습니다"}), 400

        data = request.json
        messages = data.get("messages")

        if not messages:
            return jsonify({"error": "messages가 필요합니다"}), 400

        try:
            response = llm.chat(
                messages=messages,
                max_tokens=data.get("max_tokens", 256),
                temperature=data.get("temperature", 0.7)
            )

            return jsonify({
                "response": response,
                "model": os.path.basename(llm.model_path)
            })

        except Exception as e:
            return jsonify({"error": str(e)}), 500

    @app.route("/embed", methods=["POST"])
    def embed():
        """텍스트 임베딩"""
        if not enable_embeddings or embedder is None:
            return jsonify({"error": "임베딩이 활성화되지 않았습니다"}), 400

        data = request.json
        texts = data.get("texts")

        if not texts:
            return jsonify({"error": "texts가 필요합니다"}), 400

        # Handle single string
        single_input = isinstance(texts, str)
        if single_input:
            texts = [texts]

        try:
            embeddings = embedder.encode(texts)

            # Convert to list for JSON serialization
            if single_input:
                embeddings_list = embeddings.tolist()
            else:
                embeddings_list = [emb.tolist() for emb in embeddings]

            return jsonify({
                "embeddings": embeddings_list,
                "model": embedder.model_name,
                "dimension": embedder.embedding_dim
            })

        except Exception as e:
            return jsonify({"error": str(e)}), 500

    @app.route("/search", methods=["POST"])
    def search():
        """시맨틱 검색"""
        if not enable_embeddings or embedder is None:
            return jsonify({"error": "임베딩이 활성화되지 않았습니다"}), 400

        data = request.json
        query = data.get("query")
        corpus = data.get("corpus")
        top_k = data.get("top_k", 5)

        if not query:
            return jsonify({"error": "query가 필요합니다"}), 400

        if not corpus:
            return jsonify({"error": "corpus가 필요합니다"}), 400

        try:
            results = embedder.semantic_search(
                query=query,
                corpus=corpus,
                top_k=top_k,
                return_scores=True
            )

            return jsonify({
                "results": results,
                "model": embedder.model_name
            })

        except Exception as e:
            return jsonify({"error": str(e)}), 500

    @app.route("/similarity", methods=["POST"])
    def similarity():
        """텍스트 유사도 계산"""
        if not enable_embeddings or embedder is None:
            return jsonify({"error": "임베딩이 활성화되지 않았습니다"}), 400

        data = request.json
        text1 = data.get("text1")
        text2 = data.get("text2")

        if not text1 or not text2:
            return jsonify({"error": "text1과 text2가 필요합니다"}), 400

        try:
            embeddings = embedder.encode([text1, text2])
            sim = embedder.cosine_similarity(embeddings[0], embeddings[1])

            return jsonify({
                "similarity": float(sim),
                "text1": text1,
                "text2": text2
            })

        except Exception as e:
            return jsonify({"error": str(e)}), 500

    @app.route("/cluster", methods=["POST"])
    def cluster():
        """텍스트 클러스터링"""
        if not enable_embeddings or embedder is None:
            return jsonify({"error": "임베딩이 활성화되지 않았습니다"}), 400

        data = request.json
        texts = data.get("texts")
        n_clusters = data.get("n_clusters", 5)
        method = data.get("method", "kmeans")

        if not texts:
            return jsonify({"error": "texts가 필요합니다"}), 400

        try:
            result = embedder.cluster(
                texts=texts,
                n_clusters=n_clusters,
                method=method
            )

            return jsonify({
                "clusters": result,
                "model": embedder.model_name
            })

        except Exception as e:
            return jsonify({"error": str(e)}), 500

    return app


def main():
    """메인 함수"""
    parser = argparse.ArgumentParser(
        description="로컬 AI 서버 - 인터넷 연결 없이 AI 모델 서빙"
    )

    parser.add_argument(
        "--llm-model",
        type=str,
        help="LLM 모델 경로 (.gguf 파일)"
    )

    parser.add_argument(
        "--embedding-model",
        type=str,
        default="mini",
        help="임베딩 모델 (기본: mini)"
    )

    parser.add_argument(
        "--no-llm",
        action="store_true",
        help="LLM 비활성화"
    )

    parser.add_argument(
        "--no-embeddings",
        action="store_true",
        help="임베딩 비활성화"
    )

    parser.add_argument(
        "--host",
        type=str,
        default="127.0.0.1",
        help="호스트 주소 (기본: 127.0.0.1)"
    )

    parser.add_argument(
        "--port",
        type=int,
        default=8000,
        help="포트 번호 (기본: 8000)"
    )

    parser.add_argument(
        "--verbose",
        action="store_true",
        help="상세 로그 출력"
    )

    args = parser.parse_args()

    # Validation
    enable_llm = not args.no_llm
    enable_embeddings = not args.no_embeddings

    if enable_llm and not args.llm_model:
        print("경고: --llm-model이 지정되지 않아 LLM이 비활성화됩니다.")
        print("       LLM을 사용하려면 GGUF 모델 경로를 지정하세요.")
        enable_llm = False

    if not enable_llm and not enable_embeddings:
        print("오류: LLM과 임베딩을 모두 비활성화할 수 없습니다.")
        sys.exit(1)

    # Create and run app
    print("\n" + "="*70)
    print("로컬 AI 서버 시작")
    print("="*70)

    app = create_app(
        llm_model_path=args.llm_model,
        embedding_model=args.embedding_model,
        enable_llm=enable_llm,
        enable_embeddings=enable_embeddings,
        verbose=args.verbose
    )

    print(f"\n서버 주소: http://{args.host}:{args.port}")
    print("\n사용 가능한 엔드포인트:")

    if enable_llm and args.llm_model:
        print("  - POST /generate    : 텍스트 생성")
        print("  - POST /chat        : 채팅 완성")

    if enable_embeddings:
        print("  - POST /embed       : 텍스트 임베딩")
        print("  - POST /search      : 시맨틱 검색")
        print("  - POST /similarity  : 유사도 계산")
        print("  - POST /cluster     : 텍스트 클러스터링")

    print("  - GET  /models      : 모델 정보")
    print("  - GET  /health      : 헬스 체크")

    print("\n종료: Ctrl+C")
    print("="*70 + "\n")

    # Run server
    app.run(
        host=args.host,
        port=args.port,
        debug=False
    )


if __name__ == "__main__":
    main()
