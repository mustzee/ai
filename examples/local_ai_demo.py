"""
Local AI Demo - 로컬 AI 모델 사용 예제

이 스크립트는 인터넷 연결 없이 로컬에서 실행 가능한 AI 모델들을 시연합니다.

사전 준비:
1. 의존성 설치:
   pip install llama-cpp-python sentence-transformers

2. LLM 모델 다운로드 (선택사항):
   - HuggingFace에서 GGUF 모델 다운로드
   - 예: TinyLlama, Llama-2, Mistral 등

Usage:
    # 임베딩 모델만 (모델 자동 다운로드)
    python examples/local_ai_demo.py --demo embeddings

    # LLM과 임베딩 모두 (LLM 모델 경로 필요)
    python examples/local_ai_demo.py --llm-model models/tinyllama.gguf --demo all
"""

import argparse
import sys
import os

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))


def demo_embeddings():
    """임베딩 모델 데모"""
    print("\n" + "="*70)
    print("로컬 임베딩 모델 데모")
    print("="*70)

    try:
        from src.local_embeddings import LocalEmbeddings
    except ImportError as e:
        print(f"오류: {e}")
        print("\n설치: pip install sentence-transformers")
        return

    # 1. 기본 임베딩 생성
    print("\n[1] 기본 임베딩 생성")
    print("-" * 70)

    embedder = LocalEmbeddings(model_name="mini", verbose=True)

    texts = [
        "인공지능은 정말 재미있어요",
        "AI is very interesting",
        "Machine learning is a subset of AI",
        "오늘 날씨가 좋네요",
        "딥러닝은 강력한 기술입니다"
    ]

    print(f"\n임베딩 생성 중... (텍스트 {len(texts)}개)")
    embeddings = embedder.encode(texts)
    print(f"✓ 완료! Shape: {embeddings.shape}")

    # 2. 유사도 계산
    print("\n[2] 텍스트 유사도 계산")
    print("-" * 70)

    pairs = [
        (0, 1),  # 한국어-영어 (같은 의미)
        (1, 2),  # 영어-영어 (관련 있음)
        (0, 3),  # 무관한 텍스트
        (0, 4),  # 한국어-한국어 (관련 있음)
    ]

    for i, j in pairs:
        sim = embedder.cosine_similarity(embeddings[i], embeddings[j])
        print(f"\n'{texts[i][:30]}'")
        print(f"  vs")
        print(f"'{texts[j][:30]}'")
        print(f"  → 유사도: {sim:.3f}")

    # 3. 시맨틱 검색
    print("\n[3] 시맨틱 검색")
    print("-" * 70)

    corpus = [
        "파이썬은 프로그래밍 언어입니다",
        "자바스크립트는 웹 개발에 사용됩니다",
        "머신러닝은 데이터로부터 학습합니다",
        "딥러닝은 신경망을 사용합니다",
        "강아지는 귀여운 동물입니다",
        "고양이는 집에서 키우기 좋습니다",
        "식물은 광합성을 합니다"
    ]

    queries = [
        "AI와 딥러닝에 대해 알려주세요",
        "프로그래밍 언어 추천해주세요",
        "반려동물 키우고 싶어요"
    ]

    for query in queries:
        print(f"\n쿼리: '{query}'")
        results = embedder.semantic_search(query, corpus, top_k=3)

        for idx, result in enumerate(results, 1):
            print(f"  {idx}. {result['text']} (score: {result['score']:.3f})")

    # 4. 클러스터링
    print("\n[4] 텍스트 클러스터링")
    print("-" * 70)

    documents = [
        "머신러닝 알고리즘",
        "딥러닝 신경망",
        "인공지능 개발",
        "사과와 바나나",
        "과일 샐러드",
        "건강한 음식",
        "축구 경기",
        "야구 선수",
        "스포츠 뉴스"
    ]

    print(f"\n{len(documents)}개 문서를 3개 클러스터로 분류...")
    result = embedder.cluster(documents, n_clusters=3)

    for cluster_id, texts in result['clusters'].items():
        print(f"\n클러스터 {cluster_id}:")
        for text in texts:
            print(f"  - {text}")

    # 5. 중복 검출
    print("\n[5] 중복 텍스트 검출")
    print("-" * 70)

    texts_with_duplicates = [
        "머신러닝은 강력한 도구입니다",
        "딥러닝은 신경망을 사용합니다",
        "머신러닝은 정말 강력한 도구입니다",  # 0과 유사
        "완전히 다른 내용의 텍스트입니다",
        "딥러닝은 신경망 기반입니다"  # 1과 유사
    ]

    duplicates = embedder.find_duplicates(texts_with_duplicates, threshold=0.8)

    if duplicates:
        print(f"\n{len(duplicates)}개의 유사한 쌍 발견:")
        for dup in duplicates:
            print(f"\n[{dup['index1']}] {dup['text1'][:50]}")
            print(f"[{dup['index2']}] {dup['text2'][:50]}")
            print(f"  → 유사도: {dup['similarity']:.3f}")
    else:
        print("\n중복 없음")

    print("\n" + "="*70)
    print("임베딩 데모 완료!")
    print("="*70)


def demo_llm(model_path: str):
    """LLM 모델 데모"""
    print("\n" + "="*70)
    print("로컬 LLM 데모")
    print("="*70)

    try:
        from src.local_llm import LocalLLM
    except ImportError as e:
        print(f"오류: {e}")
        print("\n설치: pip install llama-cpp-python")
        return

    if not os.path.exists(model_path):
        print(f"\n오류: 모델 파일을 찾을 수 없습니다: {model_path}")
        print("\nLLM 모델을 다운로드하세요:")
        print("1. https://huggingface.co/TheBloke 방문")
        print("2. GGUF 모델 검색 (예: TinyLlama-1.1B-Chat-v1.0-GGUF)")
        print("3. Q4_K_M 파일 다운로드")
        print("4. models/ 폴더에 저장")
        return

    # 1. 모델 로드
    print("\n[1] 모델 로드")
    print("-" * 70)

    llm = LocalLLM(
        model_path=model_path,
        n_ctx=2048,
        n_gpu_layers=0,  # CPU only (GPU 사용시 숫자 증가)
        verbose=True
    )

    # 2. 간단한 텍스트 생성
    print("\n[2] 텍스트 생성")
    print("-" * 70)

    prompts = [
        "What is artificial intelligence?",
        "Explain machine learning in simple terms.",
    ]

    for prompt in prompts:
        print(f"\n프롬프트: {prompt}")
        print("응답:", end=" ", flush=True)

        response = llm.generate(
            prompt,
            max_tokens=150,
            temperature=0.7
        )

        print(response.strip())

    # 3. 채팅
    print("\n[3] 채팅 대화")
    print("-" * 70)

    messages = [
        {"role": "system", "content": "You are a helpful AI assistant."},
        {"role": "user", "content": "Hello! Can you help me understand neural networks?"}
    ]

    print("\n대화:")
    for msg in messages:
        role = msg['role'].upper()
        print(f"{role}: {msg['content']}")

    print("\nASSISTANT:", end=" ", flush=True)
    response = llm.chat(messages, max_tokens=200)
    print(response.strip())

    # 4. 스트리밍 생성
    print("\n[4] 스트리밍 생성")
    print("-" * 70)

    prompt = "Tell me a short story about a robot:"
    print(f"\n프롬프트: {prompt}")
    print("응답: ", end="", flush=True)

    stream = llm.generate(
        prompt,
        max_tokens=150,
        temperature=0.8,
        stream=True
    )

    for token in stream:
        print(token, end="", flush=True)

    print()  # Newline

    print("\n" + "="*70)
    print("LLM 데모 완료!")
    print("="*70)


def demo_server_client():
    """서버 API 클라이언트 데모"""
    print("\n" + "="*70)
    print("로컬 AI 서버 API 클라이언트 데모")
    print("="*70)

    try:
        import requests
    except ImportError:
        print("오류: requests가 설치되지 않았습니다")
        print("설치: pip install requests")
        return

    base_url = "http://127.0.0.1:8000"

    print(f"\n서버 주소: {base_url}")
    print("(먼저 다른 터미널에서 서버를 시작하세요)")
    print("  python src/local_server.py --embedding-model mini")

    # 1. 헬스 체크
    print("\n[1] 헬스 체크")
    print("-" * 70)

    try:
        response = requests.get(f"{base_url}/health", timeout=5)
        if response.status_code == 200:
            print("✓ 서버 정상 작동")
            print(f"  응답: {response.json()}")
        else:
            print("✗ 서버 오류")
            return
    except Exception as e:
        print(f"✗ 서버 연결 실패: {e}")
        print("\n서버를 시작해주세요:")
        print("  python src/local_server.py --embedding-model mini")
        return

    # 2. 모델 정보
    print("\n[2] 모델 정보")
    print("-" * 70)

    response = requests.get(f"{base_url}/models")
    models = response.json()
    print(f"로드된 모델: {list(models.keys())}")
    for name, info in models.items():
        print(f"\n{name}:")
        for key, value in info.items():
            print(f"  {key}: {value}")

    # 3. 임베딩 생성
    print("\n[3] 임베딩 생성")
    print("-" * 70)

    texts = ["Hello world", "AI is amazing"]
    response = requests.post(
        f"{base_url}/embed",
        json={"texts": texts}
    )

    if response.status_code == 200:
        result = response.json()
        print(f"✓ 임베딩 생성 완료")
        print(f"  모델: {result['model']}")
        print(f"  차원: {result['dimension']}")
        print(f"  생성된 임베딩 개수: {len(result['embeddings'])}")
    else:
        print(f"✗ 오류: {response.json()}")

    # 4. 유사도 계산
    print("\n[4] 유사도 계산")
    print("-" * 70)

    response = requests.post(
        f"{base_url}/similarity",
        json={
            "text1": "I love machine learning",
            "text2": "AI and deep learning are great"
        }
    )

    if response.status_code == 200:
        result = response.json()
        print(f"✓ 유사도: {result['similarity']:.3f}")
    else:
        print(f"✗ 오류: {response.json()}")

    # 5. 시맨틱 검색
    print("\n[5] 시맨틱 검색")
    print("-" * 70)

    response = requests.post(
        f"{base_url}/search",
        json={
            "query": "programming languages",
            "corpus": [
                "Python is a programming language",
                "Dogs are cute animals",
                "JavaScript is used for web development",
                "Cats are popular pets"
            ],
            "top_k": 2
        }
    )

    if response.status_code == 200:
        result = response.json()
        print("✓ 검색 결과:")
        for idx, item in enumerate(result['results'], 1):
            print(f"  {idx}. {item['text']} (score: {item['score']:.3f})")
    else:
        print(f"✗ 오류: {response.json()}")

    print("\n" + "="*70)
    print("API 클라이언트 데모 완료!")
    print("="*70)


def main():
    """메인 함수"""
    parser = argparse.ArgumentParser(
        description="로컬 AI 모델 데모"
    )

    parser.add_argument(
        "--demo",
        choices=["embeddings", "llm", "server", "all"],
        default="embeddings",
        help="실행할 데모 (기본: embeddings)"
    )

    parser.add_argument(
        "--llm-model",
        type=str,
        help="LLM 모델 경로 (llm/all 데모에 필요)"
    )

    args = parser.parse_args()

    print("\n" + "="*70)
    print("로컬 AI 데모 - 인터넷 연결 없이 AI 모델 사용하기")
    print("="*70)

    # Run demos
    if args.demo in ["embeddings", "all"]:
        demo_embeddings()

    if args.demo in ["llm", "all"]:
        if not args.llm_model:
            print("\n경고: --llm-model이 지정되지 않아 LLM 데모를 건너뜁니다.")
            print("      LLM 데모를 실행하려면 GGUF 모델 경로를 지정하세요.")
            print("\n예:")
            print("  python examples/local_ai_demo.py --demo llm \\")
            print("    --llm-model models/tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf")
        else:
            demo_llm(args.llm_model)

    if args.demo == "server":
        demo_server_client()

    print("\n모든 데모 완료! 🎉")


if __name__ == "__main__":
    main()
