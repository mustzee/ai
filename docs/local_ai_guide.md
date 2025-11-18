# 로컬 AI 모델 가이드

**인터넷 연결 없이 AI 모델을 로컬에서 실행하는 완벽한 가이드**

---

## 📋 목차

1. [개요](#개요)
2. [설치](#설치)
3. [로컬 LLM 사용하기](#로컬-llm-사용하기)
4. [로컬 임베딩 사용하기](#로컬-임베딩-사용하기)
5. [로컬 AI 서버](#로컬-ai-서버)
6. [실전 예제](#실전-예제)
7. [문제 해결](#문제-해결)

---

## 개요

이 프로젝트는 **인터넷 연결 없이** 로컬에서 실행 가능한 AI 모델 서빙 시스템을 제공합니다.

### 🎯 주요 기능

- **로컬 LLM**: llama.cpp 기반 대규모 언어 모델 실행
- **로컬 임베딩**: sentence-transformers 기반 텍스트 임베딩
- **REST API 서버**: Flask 기반 간단한 API 서버
- **완전 오프라인**: 모델 다운로드 후 인터넷 불필요

### 💡 사용 사례

- 개인 정보 보호가 중요한 애플리케이션
- 인터넷 연결이 불안정한 환경
- 저비용으로 AI 서비스 구축
- 학습 및 실험용 AI 시스템

---

## 설치

### 1. 기본 의존성 설치

```bash
# 기본 패키지
pip install -r requirements.txt

# 또는 개별 설치
pip install numpy matplotlib scikit-learn
```

### 2. 로컬 LLM 지원 (선택사항)

```bash
# CPU만 사용
pip install llama-cpp-python

# GPU 지원 (CUDA)
CMAKE_ARGS="-DLLAMA_CUBLAS=on" pip install llama-cpp-python

# GPU 지원 (Metal - Mac)
CMAKE_ARGS="-DLLAMA_METAL=on" pip install llama-cpp-python
```

### 3. 로컬 임베딩 지원

```bash
pip install sentence-transformers
```

### 4. API 서버 지원 (선택사항)

```bash
pip install flask requests
```

---

## 로컬 LLM 사용하기

### 모델 다운로드

#### 추천 모델

| 모델 | 크기 | 메모리 | 용도 | 링크 |
|------|------|--------|------|------|
| **TinyLlama 1.1B** | ~700MB | 2GB | 실험/학습용 | [다운로드](https://huggingface.co/TheBloke/TinyLlama-1.1B-Chat-v1.0-GGUF) |
| **Mistral 7B** | ~4GB | 8GB | 일반 용도 | [다운로드](https://huggingface.co/TheBloke/Mistral-7B-Instruct-v0.2-GGUF) |
| **Llama 2 7B** | ~4GB | 8GB | 고품질 | [다운로드](https://huggingface.co/TheBloke/Llama-2-7B-Chat-GGUF) |
| **Llama 2 13B** | ~7GB | 16GB | 최고 품질 | [다운로드](https://huggingface.co/TheBloke/Llama-2-13B-Chat-GGUF) |

#### 다운로드 방법

```bash
# 1. models 폴더 생성
mkdir -p models

# 2. wget으로 다운로드
cd models
wget https://huggingface.co/TheBloke/TinyLlama-1.1B-Chat-v1.0-GGUF/resolve/main/tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf

# 또는 브라우저에서 직접 다운로드
```

#### 양자화 수준 선택

| 양자화 | 파일 크기 | 품질 | 추천 |
|--------|-----------|------|------|
| Q3_K_M | 작음 | 낮음 | ❌ |
| **Q4_K_M** | 중간 | 좋음 | ✅ 추천 |
| Q5_K_M | 큼 | 매우 좋음 | ✅ |
| Q6_K | 매우 큼 | 최고 | ⚠️ 고사양 |

### 기본 사용법

```python
from src.local_llm import LocalLLM

# 1. 모델 로드
llm = LocalLLM(
    model_path="models/tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf",
    n_ctx=2048,        # 컨텍스트 길이
    n_gpu_layers=0,    # GPU 레이어 수 (0 = CPU only)
    verbose=True
)

# 2. 텍스트 생성
response = llm.generate(
    prompt="What is artificial intelligence?",
    max_tokens=200,
    temperature=0.7
)
print(response)

# 3. 채팅
messages = [
    {"role": "system", "content": "You are a helpful assistant."},
    {"role": "user", "content": "Explain machine learning"}
]
response = llm.chat(messages, max_tokens=200)
print(response)

# 4. 스트리밍 생성
stream = llm.generate(
    "Tell me a story:",
    max_tokens=200,
    stream=True
)
for token in stream:
    print(token, end="", flush=True)
```

### 고급 설정

```python
# GPU 가속 (CUDA)
llm = LocalLLM(
    model_path="models/llama-2-7b-chat.Q4_K_M.gguf",
    n_gpu_layers=35,   # GPU로 오프로드할 레이어 수
    n_ctx=4096,        # 더 긴 컨텍스트
    verbose=True
)

# 파라미터 조정
response = llm.generate(
    prompt="Creative writing:",
    max_tokens=500,
    temperature=0.9,      # 더 창의적 (0.0 ~ 1.0)
    top_p=0.95,           # Nucleus sampling
    top_k=50,             # Top-K sampling
    repeat_penalty=1.2    # 반복 억제
)
```

### 여러 모델 관리

```python
from src.local_llm import LocalLLMServer

# 서버 생성
server = LocalLLMServer(verbose=True)

# 여러 모델 로드
server.load_model(
    name="tiny",
    model_path="models/tinyllama.gguf"
)
server.load_model(
    name="mistral",
    model_path="models/mistral-7b.gguf"
)

# 모델 사용
response = server.generate("tiny", "Hello!")
response = server.generate("mistral", "Explain AI")

# 로드된 모델 확인
print(server.list_models())  # ['tiny', 'mistral']
```

---

## 로컬 임베딩 사용하기

### 모델 선택

| 별칭 | 모델 이름 | 크기 | 차원 | 용도 |
|------|-----------|------|------|------|
| **mini** | all-MiniLM-L6-v2 | 80MB | 384 | 빠른 일반 용도 ✅ |
| **multilingual** | paraphrase-multilingual-MiniLM-L12-v2 | 420MB | 384 | 다국어 지원 |
| **korean** | jhgan/ko-sroberta-multitask | 300MB | 768 | 한국어 특화 🇰🇷 |
| **base** | all-mpnet-base-v2 | 420MB | 768 | 고품질 영어 |
| **large** | sentence-t5-base | 220MB | 768 | 최고 품질 |

### 기본 사용법

```python
from src.local_embeddings import LocalEmbeddings

# 1. 모델 로드 (처음 실행시 자동 다운로드)
embedder = LocalEmbeddings(
    model_name="mini",  # 또는 "korean", "multilingual" 등
    device="cpu",       # 또는 "cuda", "mps"
    verbose=True
)

# 2. 임베딩 생성
texts = [
    "인공지능은 흥미로운 분야입니다",
    "Machine learning is fascinating",
    "딥러닝은 강력한 기술입니다"
]
embeddings = embedder.encode(texts)
print(embeddings.shape)  # (3, 384)

# 3. 유사도 계산
similarity = embedder.cosine_similarity(
    embeddings[0],
    embeddings[1]
)
print(f"유사도: {similarity:.3f}")  # 0.0 ~ 1.0
```

### 시맨틱 검색

```python
# 쿼리와 가장 유사한 문서 찾기
query = "프로그래밍 언어 추천"
corpus = [
    "파이썬은 배우기 쉬운 언어입니다",
    "자바스크립트는 웹 개발에 사용됩니다",
    "고양이는 귀여운 동물입니다",
    "C++는 성능이 뛰어납니다"
]

results = embedder.semantic_search(
    query=query,
    corpus=corpus,
    top_k=2
)

for r in results:
    print(f"{r['text']} (score: {r['score']:.3f})")
```

**출력:**
```
파이썬은 배우기 쉬운 언어입니다 (score: 0.874)
자바스크립트는 웹 개발에 사용됩니다 (score: 0.821)
```

### 텍스트 클러스터링

```python
documents = [
    "머신러닝 알고리즘",
    "딥러닝 신경망",
    "AI 개발",
    "사과와 바나나",
    "과일 샐러드",
    "축구 경기",
    "야구 선수"
]

result = embedder.cluster(
    texts=documents,
    n_clusters=3,
    method="kmeans"
)

for cluster_id, texts in result['clusters'].items():
    print(f"\n클러스터 {cluster_id}:")
    for text in texts:
        print(f"  - {text}")
```

**출력:**
```
클러스터 0:
  - 머신러닝 알고리즘
  - 딥러닝 신경망
  - AI 개발

클러스터 1:
  - 사과와 바나나
  - 과일 샐러드

클러스터 2:
  - 축구 경기
  - 야구 선수
```

### 중복 검출

```python
texts = [
    "머신러닝은 강력한 도구입니다",
    "딥러닝은 신경망을 사용합니다",
    "머신러닝은 정말 강력한 도구입니다",  # 0과 유사
    "완전히 다른 내용입니다"
]

duplicates = embedder.find_duplicates(
    texts=texts,
    threshold=0.85
)

for dup in duplicates:
    print(f"유사: {dup['text1']}")
    print(f"  ↔ {dup['text2']}")
    print(f"  유사도: {dup['similarity']:.3f}\n")
```

### 임베딩 캐시 (성능 최적화)

```python
from src.local_embeddings import EmbeddingCache

# 캐시 사용
cache = EmbeddingCache(
    embedder=embedder,
    max_cache_size=10000
)

# 동일한 텍스트는 캐시에서 가져옴
embedding1 = cache.encode("Hello world")  # 계산
embedding2 = cache.encode("Hello world")  # 캐시에서 가져옴 (빠름!)

print(f"캐시 크기: {cache.get_cache_size()}")
```

---

## 로컬 AI 서버

### 서버 시작

```bash
# 임베딩 모델만
python src/local_server.py \
  --embedding-model mini \
  --port 8000

# LLM + 임베딩
python src/local_server.py \
  --llm-model models/tinyllama.gguf \
  --embedding-model mini \
  --port 8000

# 한국어 임베딩 사용
python src/local_server.py \
  --embedding-model korean \
  --port 8000
```

### API 엔드포인트

#### 1. 헬스 체크

```bash
curl http://localhost:8000/health
```

**응답:**
```json
{
  "status": "healthy",
  "llm_enabled": true,
  "embeddings_enabled": true
}
```

#### 2. 모델 정보

```bash
curl http://localhost:8000/models
```

#### 3. 텍스트 생성 (LLM)

```bash
curl -X POST http://localhost:8000/generate \
  -H "Content-Type: application/json" \
  -d '{
    "prompt": "What is AI?",
    "max_tokens": 200,
    "temperature": 0.7
  }'
```

#### 4. 채팅 (LLM)

```bash
curl -X POST http://localhost:8000/chat \
  -H "Content-Type: application/json" \
  -d '{
    "messages": [
      {"role": "system", "content": "You are helpful."},
      {"role": "user", "content": "Explain machine learning"}
    ],
    "max_tokens": 200
  }'
```

#### 5. 임베딩 생성

```bash
curl -X POST http://localhost:8000/embed \
  -H "Content-Type: application/json" \
  -d '{
    "texts": ["Hello world", "AI is great"]
  }'
```

**응답:**
```json
{
  "embeddings": [[0.123, -0.456, ...], [0.789, 0.234, ...]],
  "model": "all-MiniLM-L6-v2",
  "dimension": 384
}
```

#### 6. 유사도 계산

```bash
curl -X POST http://localhost:8000/similarity \
  -H "Content-Type: application/json" \
  -d '{
    "text1": "I love programming",
    "text2": "Coding is fun"
  }'
```

**응답:**
```json
{
  "similarity": 0.827,
  "text1": "I love programming",
  "text2": "Coding is fun"
}
```

#### 7. 시맨틱 검색

```bash
curl -X POST http://localhost:8000/search \
  -H "Content-Type: application/json" \
  -d '{
    "query": "programming languages",
    "corpus": [
      "Python is a programming language",
      "Dogs are cute",
      "JavaScript for web development"
    ],
    "top_k": 2
  }'
```

#### 8. 클러스터링

```bash
curl -X POST http://localhost:8000/cluster \
  -H "Content-Type: application/json" \
  -d '{
    "texts": ["ML", "AI", "cat", "dog", "soccer", "baseball"],
    "n_clusters": 3,
    "method": "kmeans"
  }'
```

### Python 클라이언트

```python
import requests

BASE_URL = "http://localhost:8000"

# 임베딩 생성
response = requests.post(
    f"{BASE_URL}/embed",
    json={"texts": ["Hello", "World"]}
)
embeddings = response.json()['embeddings']

# 유사도 계산
response = requests.post(
    f"{BASE_URL}/similarity",
    json={
        "text1": "AI is cool",
        "text2": "Machine learning is awesome"
    }
)
similarity = response.json()['similarity']
print(f"유사도: {similarity:.3f}")

# 시맨틱 검색
response = requests.post(
    f"{BASE_URL}/search",
    json={
        "query": "프로그래밍",
        "corpus": ["파이썬", "강아지", "자바스크립트"],
        "top_k": 2
    }
)
results = response.json()['results']
for r in results:
    print(f"{r['text']}: {r['score']:.3f}")
```

---

## 실전 예제

### 예제 1: RAG (Retrieval Augmented Generation)

```python
from src.local_llm import LocalLLM
from src.local_embeddings import LocalEmbeddings

# 1. 지식 베이스
knowledge_base = [
    "파이썬은 1991년 귀도 반 로섬이 만든 프로그래밍 언어입니다.",
    "머신러닝은 데이터로부터 패턴을 학습하는 AI 기술입니다.",
    "딥러닝은 인공 신경망을 사용하는 머신러닝의 한 분야입니다.",
    "NumPy는 파이썬의 수치 계산 라이브러리입니다."
]

# 2. 임베딩 모델 로드
embedder = LocalEmbeddings(model_name="korean")

# 3. LLM 로드
llm = LocalLLM(model_path="models/llama-2-7b.gguf")

# 4. 질문
question = "딥러닝이 뭐야?"

# 5. 관련 문서 검색
results = embedder.semantic_search(
    query=question,
    corpus=knowledge_base,
    top_k=2
)

# 6. 컨텍스트 구성
context = "\n".join([r['text'] for r in results])

# 7. 프롬프트 생성
prompt = f"""다음 정보를 바탕으로 질문에 답하세요:

정보:
{context}

질문: {question}
답변:"""

# 8. 답변 생성
answer = llm.generate(prompt, max_tokens=200)
print(answer)
```

### 예제 2: 문서 분류

```python
from src.local_embeddings import LocalEmbeddings
import numpy as np

# 1. 임베딩 모델
embedder = LocalEmbeddings(model_name="mini")

# 2. 카테고리 정의
categories = {
    "기술": ["프로그래밍", "AI", "소프트웨어"],
    "스포츠": ["축구", "야구", "운동"],
    "음식": ["요리", "레시피", "맛집"]
}

# 3. 카테고리 임베딩
category_embeddings = {}
for cat, keywords in categories.items():
    embs = embedder.encode(keywords)
    category_embeddings[cat] = np.mean(embs, axis=0)

# 4. 문서 분류
documents = [
    "파이썬으로 웹 개발하기",
    "맛있는 파스타 만드는 법",
    "월드컵 축구 경기 결과"
]

for doc in documents:
    doc_emb = embedder.encode(doc)

    # 각 카테고리와 유사도 계산
    similarities = {}
    for cat, cat_emb in category_embeddings.items():
        sim = embedder.cosine_similarity(doc_emb, cat_emb)
        similarities[cat] = sim

    # 가장 유사한 카테고리
    best_cat = max(similarities, key=similarities.get)
    print(f"'{doc}' → {best_cat} ({similarities[best_cat]:.3f})")
```

### 예제 3: 챗봇 (기존 TinyAI + 로컬 LLM)

```python
from src.chatbot import TinyAIAssistant
from src.local_llm import LocalLLM

# 1. 기존 의도 분류 챗봇
tiny_chatbot = TinyAIAssistant()
tiny_chatbot.load_intents("data/intents.json")
tiny_chatbot.train()

# 2. LLM 백엔드
llm = LocalLLM(model_path="models/mistral-7b.gguf")

# 3. 하이브리드 응답
def hybrid_response(user_input):
    # 의도 분류
    intent, confidence = tiny_chatbot.classifier.predict(user_input)

    if confidence > 0.5:
        # 높은 신뢰도: 템플릿 응답
        return tiny_chatbot.generate_response(intent)
    else:
        # 낮은 신뢰도: LLM 사용
        messages = [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": user_input}
        ]
        return llm.chat(messages, max_tokens=200)

# 사용
while True:
    user_input = input("You: ")
    if user_input.lower() in ['quit', 'exit']:
        break
    response = hybrid_response(user_input)
    print(f"Bot: {response}")
```

---

## 문제 해결

### LLM 관련

#### Q: "모델이 너무 느려요"
**A:** 다음을 시도해보세요:
1. GPU 사용: `n_gpu_layers=35` 설정
2. 더 작은 모델 사용 (TinyLlama)
3. 양자화 수준 낮추기 (Q4_K_M → Q3_K_M)
4. `n_ctx` 줄이기 (4096 → 2048)

#### Q: "메모리 부족 오류"
**A:**
1. 더 작은 모델 사용
2. `n_ctx` 줄이기
3. 낮은 양자화 모델 사용 (Q3_K_M)

#### Q: "모델 로딩 실패"
**A:**
1. 모델 파일 경로 확인
2. GGUF 형식인지 확인
3. `llama-cpp-python` 재설치

```bash
pip uninstall llama-cpp-python
pip install llama-cpp-python --no-cache-dir
```

### 임베딩 관련

#### Q: "첫 실행시 모델 다운로드가 안돼요"
**A:**
1. 인터넷 연결 확인 (첫 다운로드시 필요)
2. 수동 다운로드:
```python
from sentence_transformers import SentenceTransformer
model = SentenceTransformer("all-MiniLM-L6-v2")
model.save("./local_models/mini")
```

#### Q: "한국어 성능이 안좋아요"
**A:**
- 한국어 특화 모델 사용:
```python
embedder = LocalEmbeddings(model_name="korean")
```

### 서버 관련

#### Q: "서버가 시작 안돼요"
**A:**
1. 포트 충돌 확인: `--port 8001`
2. Flask 설치 확인: `pip install flask`
3. 방화벽 확인

#### Q: "요청이 너무 느려요"
**A:**
1. 더 작은 모델 사용
2. 배치 처리
3. 캐싱 활용

---

## 성능 비교

### LLM 모델

| 모델 | 크기 | 메모리 | 속도 (tokens/s) | 품질 |
|------|------|--------|-----------------|------|
| TinyLlama 1.1B | 700MB | 2GB | ~30 | ⭐⭐⭐ |
| Mistral 7B | 4GB | 8GB | ~15 | ⭐⭐⭐⭐⭐ |
| Llama 2 7B | 4GB | 8GB | ~12 | ⭐⭐⭐⭐ |
| Llama 2 13B | 7GB | 16GB | ~8 | ⭐⭐⭐⭐⭐ |

### 임베딩 모델

| 모델 | 크기 | 속도 (docs/s) | 품질 | 다국어 |
|------|------|---------------|------|--------|
| mini | 80MB | ~1000 | ⭐⭐⭐ | ❌ |
| multilingual | 420MB | ~500 | ⭐⭐⭐⭐ | ✅ |
| korean | 300MB | ~400 | ⭐⭐⭐⭐ | 🇰🇷 |
| base | 420MB | ~400 | ⭐⭐⭐⭐⭐ | ❌ |

---

## 라이선스 및 리소스

### 모델 라이선스

- **Llama 2**: Meta의 Llama 2 Community License
- **Mistral**: Apache 2.0
- **Sentence-Transformers**: Apache 2.0

### 추가 리소스

- [llama.cpp GitHub](https://github.com/ggerganov/llama.cpp)
- [sentence-transformers 문서](https://www.sbert.net/)
- [HuggingFace GGUF 모델](https://huggingface.co/TheBloke)
- [Awesome Local LLMs](https://github.com/janhq/awesome-local-llms)

---

## 다음 단계

1. **벡터 데이터베이스 통합**: FAISS, ChromaDB
2. **웹 UI**: Gradio, Streamlit
3. **멀티모달**: Vision models (CLIP, LLaVA)
4. **파인튜닝**: LoRA, QLoRA
5. **프로덕션 배포**: Docker, K8s

---

**만든이**: Tiny AI Project
**버전**: 1.0.0
**마지막 업데이트**: 2024
