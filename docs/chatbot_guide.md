# Tiny AI 챗봇 가이드

친절하고 논리적인 대화형 AI 어시스턴트 구현 가이드

## 📖 목차

1. [개요](#개요)
2. [아키텍처](#아키텍처)
3. [핵심 구성요소](#핵심-구성요소)
4. [사용 방법](#사용-방법)
5. [커스터마이징](#커스터마이징)
6. [성능 튜닝](#성능-튜닝)

---

## 개요

Tiny AI 챗봇은 신경망 기반의 의도 분류와 템플릿 기반 응답 생성을 결합한 대화형 AI입니다.

### 특징

- 🧠 **신경망 기반 의도 분류**: MLP를 사용한 사용자 의도 파악
- 📝 **자연어 처리**: 토크나이저, TF-IDF 벡터화
- 💬 **친절한 톤앤매너**: 공감적이고 논리적인 응답
- 🎯 **높은 정확도**: 99% 이상의 의도 분류 정확도
- 🔧 **쉬운 확장**: JSON 파일로 새로운 의도 추가

---

## 아키텍처

### 전체 흐름

```
사용자 입력
    ↓
텍스트 전처리 (토크나이징)
    ↓
벡터화 (TF-IDF)
    ↓
의도 분류 (신경망)
    ↓
응답 생성 (템플릿)
    ↓
챗봇 응답
```

### 구성요소

```
TinyAIAssistant
├── IntentClassifier
│   ├── Tokenizer (토큰화)
│   ├── TfidfVectorizer (벡터화)
│   └── MLP (의도 분류)
└── ResponseGenerator
    └── Intents DB (응답 템플릿)
```

---

## 핵심 구성요소

### 1. 텍스트 전처리

#### Tokenizer

사용자 입력을 단어 단위로 분리합니다.

```python
from src.text_processing import Tokenizer

tokenizer = Tokenizer(max_words=500)

# 학습
texts = ["안녕하세요", "도와주세요", "감사합니다"]
tokenizer.fit(texts)

# 토크나이징
tokens = tokenizer.tokenize("안녕하세요 반갑습니다")
# ['안녕하세요', '반갑습니다']

# 시퀀스 변환
sequences = tokenizer.texts_to_sequences(["안녕하세요"])
# [[2]]  (인덱스)
```

#### TF-IDF Vectorizer

텍스트를 수치 벡터로 변환합니다.

```python
from src.text_processing import TfidfVectorizer

vectorizer = TfidfVectorizer(tokenizer)
vectorizer.fit(texts)

# 벡터화
vectors = vectorizer.transform(["안녕하세요"])
# [[0.0, 0.5, 0.0, ...]]
```

**TF-IDF란?**
- TF (Term Frequency): 단어 빈도
- IDF (Inverse Document Frequency): 역문서 빈도
- 중요한 단어에 높은 가중치 부여

### 2. 의도 분류

#### IntentClassifier

사용자 입력의 의도를 분류합니다.

```python
from src.chatbot import IntentClassifier

# 의도 정의
intents = {
    "greeting": {
        "patterns": ["안녕", "안녕하세요", "반가워"],
        "responses": ["안녕하세요!", "반갑습니다!"]
    }
}

# 분류기 생성 및 학습
classifier = IntentClassifier(intents)
classifier.train(epochs=1000)

# 예측
intent, confidence = classifier.predict("안녕하세요")
print(f"의도: {intent}, 신뢰도: {confidence:.2%}")
# 의도: greeting, 신뢰도: 96.00%
```

**학습 과정:**
```
Epoch 1: Accuracy 8%
Epoch 100: Accuracy 24%
Epoch 200: Accuracy 86%
Epoch 300: Accuracy 99%
```

**신경망 구조:**
```
입력층 (133) → 은닉층1 (64) → 은닉층2 (32) → 출력층 (14)
```

### 3. 응답 생성

#### ResponseGenerator

의도에 맞는 응답을 생성합니다.

```python
from src.chatbot import ResponseGenerator

generator = ResponseGenerator(intents)

# 응답 생성
response = generator.generate("greeting")
# "안녕하세요!" 또는 "반갑습니다!" (랜덤 선택)

# 컨텍스트 활용
response = generator.generate("greeting", context={"name": "철수"})
```

---

## 사용 방법

### 기본 사용

```python
from src.chatbot import TinyAIAssistant

# 1. 초기화
assistant = TinyAIAssistant(intents_file='data/intents.json')

# 2. 학습
assistant.train(epochs=1000)

# 3. 대화
response = assistant.chat("안녕하세요!")
print(response)

response = assistant.chat("공부 도와줘")
print(response)

response = assistant.chat("고마워")
print(response)
```

### CLI 인터페이스

```bash
python examples/chatbot_demo.py
```

**대화 예시:**
```
👤 당신: 안녕!
🤖 AI: 안녕하세요! 오늘 하루는 어떠신가요?

👤 당신: 기분이 안 좋아
🤖 AI: 힘든 시기를 겪고 계시는군요. 하지만 이 또한 지나갈 거예요.

👤 당신: 고마워
🤖 AI: 천만에요! 도움이 되었다니 기쁩니다.
```

### 대화 기록 관리

```python
# 기록 조회
history = assistant.get_conversation_history()
for msg in history:
    print(f"{msg['role']}: {msg['content']}")

# 기록 초기화
assistant.clear_history()
```

### 모델 저장/로드

```python
# 모델 저장
assistant.save_model('my_chatbot.npz')

# 모델 로드
assistant.load_model('my_chatbot.npz')
```

---

## 커스터마이징

### 새로운 의도 추가

`data/intents.json` 파일을 수정합니다:

```json
{
  "intents": {
    "new_intent": {
      "patterns": [
        "패턴1",
        "패턴2",
        "패턴3"
      ],
      "responses": [
        "응답1",
        "응답2",
        "응답3"
      ]
    }
  }
}
```

**예시: 날씨 질문 추가**

```json
{
  "weather": {
    "patterns": [
      "날씨 어때",
      "오늘 날씨",
      "비 와",
      "맑아"
    ],
    "responses": [
      "실시간 날씨 정보는 제공할 수 없어요.",
      "날씨 앱을 확인해 보시는 건 어떨까요?",
      "기상청 웹사이트에서 확인하실 수 있어요."
    ]
  }
}
```

### 응답 템플릿 변수

컨텍스트 변수를 사용할 수 있습니다:

```json
{
  "greeting": {
    "patterns": ["안녕"],
    "responses": [
      "안녕하세요, {name}님!",
      "{name}님, 반갑습니다!"
    ]
  }
}
```

```python
response = assistant.chat("안녕", context={"name": "철수"})
# "안녕하세요, 철수님!"
```

### 톤앤매너 조정

응답 문구를 수정하여 톤을 조정할 수 있습니다:

**공식적인 톤:**
```json
"responses": [
  "안녕하십니까. 무엇을 도와드릴까요?",
  "반갑습니다. 문의사항을 말씀해 주시기 바랍니다."
]
```

**친근한 톤:**
```json
"responses": [
  "안녕! 오늘 뭐 도와줄까?",
  "반가워! 무슨 일이야?"
]
```

**공감적인 톤:**
```json
"responses": [
  "힘든 시간을 보내고 계시는군요. 함께 해결책을 찾아봅시다.",
  "그런 기분이 드는 것은 충분히 이해됩니다. 어떻게 도와드릴까요?"
]
```

---

## 성능 튜닝

### 하이퍼파라미터 조정

#### 학습 에포크

```python
# 빠른 학습 (낮은 정확도)
assistant.train(epochs=100)

# 균형잡힌 학습 (권장)
assistant.train(epochs=1000)

# 철저한 학습 (높은 정확도)
assistant.train(epochs=2000)
```

#### 학습률

```python
classifier.train(epochs=1000, learning_rate=0.1)  # 느린 학습
classifier.train(epochs=1000, learning_rate=0.3)  # 빠른 학습 (권장)
classifier.train(epochs=1000, learning_rate=0.5)  # 매우 빠른 학습
```

#### 신뢰도 임계값

```python
# 낮은 임계값 (더 많은 의도 분류, 낮은 정확도)
intent, conf = classifier.predict(text, threshold=0.2)

# 중간 임계값 (권장)
intent, conf = classifier.predict(text, threshold=0.25)

# 높은 임계값 (보수적 분류, 높은 정확도)
intent, conf = classifier.predict(text, threshold=0.5)
```

### 신경망 구조 조정

```python
# 작은 모델 (빠름, 낮은 정확도)
MLP(layers=[input_size, 32, 16, num_intents])

# 중간 모델 (권장)
MLP(layers=[input_size, 64, 32, num_intents])

# 큰 모델 (느림, 높은 정확도)
MLP(layers=[input_size, 128, 64, 32, num_intents])
```

### 데이터 증강

더 많은 패턴을 추가하여 정확도를 높입니다:

```json
{
  "greeting": {
    "patterns": [
      "안녕",
      "안녕하세요",
      "하이",
      "헬로",
      "반가워",
      "좋은 아침",
      "좋은 저녁",
      "처음 뵙겠습니다",
      "만나서 반가워",
      "hi",
      "hello",
      "hey"
    ]
  }
}
```

**권장 패턴 수:**
- 최소: 5개
- 권장: 8-12개
- 이상적: 15-20개

---

## 실전 팁

### 1. 의도 설계

**좋은 의도 설계:**
- 명확하고 구분되는 의도
- 충분한 패턴 예시
- 다양한 표현 방식

**나쁜 의도 설계:**
- 모호하고 겹치는 의도
- 패턴이 너무 적음
- 단조로운 표현

### 2. 응답 작성

**좋은 응답:**
- 자연스러운 대화체
- 공감적인 표현
- 구체적인 도움 제공

```python
"힘들어하시는군요. 한 걸음씩 천천히 해결해 봅시다. 구체적으로 어떤 부분이 어려우신가요?"
```

**나쁜 응답:**
- 기계적인 표현
- 너무 짧거나 추상적

```python
"오류입니다."
```

### 3. 테스트

```python
# 다양한 입력 테스트
test_cases = [
    ("안녕", "greeting"),
    ("고마워", "thanks"),
    ("도와줘", "help_request"),
]

for text, expected in test_cases:
    intent, conf = classifier.predict(text)
    assert intent == expected, f"Failed: {text}"
    print(f"✓ {text} → {intent} ({conf:.2%})")
```

---

## 문제 해결

### 낮은 정확도

**원인:**
- 학습 데이터 부족
- 에포크 수 부족
- 학습률 문제

**해결:**
```python
# 1. 더 많은 패턴 추가
# 2. 에포크 증가
assistant.train(epochs=2000)
# 3. 학습률 조정
classifier.train(epochs=1000, learning_rate=0.3)
```

### 잘못된 의도 분류

**원인:**
- 의도 간 패턴이 유사함
- 임계값이 너무 낮음

**해결:**
```python
# 1. 의도를 더 명확하게 구분
# 2. 임계값 상향
intent, conf = classifier.predict(text, threshold=0.4)
```

### 느린 학습

**원인:**
- 모델이 너무 큼
- 배치 크기가 작음

**해결:**
```python
# 1. 모델 크기 축소
MLP(layers=[input_size, 32, 16, num_intents])
# 2. 배치 크기 증가
model.train(X, y, batch_size=16)
```

---

## 참고 자료

### 관련 파일

- `src/chatbot.py` - 챗봇 메인 로직
- `src/text_processing.py` - 텍스트 전처리
- `data/intents.json` - 의도 정의
- `examples/chatbot_demo.py` - 대화형 데모
- `examples/chatbot_test.py` - 테스트 스크립트

### 확장 아이디어

1. **실시간 정보 통합**
   - API 호출로 날씨, 시간 등 제공

2. **대화 컨텍스트 관리**
   - 이전 대화 내용 참조

3. **감정 분석**
   - 사용자 감정 파악 및 대응

4. **다국어 지원**
   - 여러 언어로 확장

5. **음성 인터페이스**
   - STT/TTS 통합
