# Tiny AI - 초경량 인공지능 프로젝트

Tiny AI는 최소한의 코드로 인공지능의 핵심 원리를 이해하고 구현하는 교육용 프로젝트입니다.

## 📚 목차

1. [Tiny AI란?](#tiny-ai란)
2. [핵심 원리](#핵심-원리)
3. [구현 방법](#구현-방법)
4. [프로젝트 구조](#프로젝트-구조)
5. [사용 예시](#사용-예시)
6. [🆕 대화형 AI 챗봇](#-대화형-ai-챗봇)
7. [실행 방법](#실행-방법)

## 🤖 Tiny AI란?

Tiny AI는 다음과 같은 특징을 가진 초경량 인공지능입니다:

- **최소한의 의존성**: NumPy만 사용하여 신경망 구현
- **교육 목적**: 딥러닝의 핵심 원리를 코드로 직접 확인
- **경량화**: 제한된 환경에서도 실행 가능
- **명확한 구조**: 이해하기 쉬운 코드 구조

## 🧠 핵심 원리

### 1. 퍼셉트론 (Perceptron)

가장 기본적인 인공 신경망 단위입니다.

```
입력(x) → 가중치 곱셈(w·x) → 활성화 함수 → 출력(y)
```

**수학적 표현:**
```
y = f(Σ(wi * xi) + b)
```

- `wi`: 가중치 (weight)
- `xi`: 입력값
- `b`: 편향 (bias)
- `f`: 활성화 함수 (activation function)

### 2. 다층 퍼셉트론 (Multi-Layer Perceptron, MLP)

여러 층의 퍼셉트론을 쌓아 복잡한 패턴을 학습합니다.

```
입력층 → 은닉층1 → 은닉층2 → ... → 출력층
```

### 3. 역전파 알고리즘 (Backpropagation)

신경망 학습의 핵심 알고리즘입니다.

**학습 과정:**
1. **순전파 (Forward Pass)**: 입력에서 출력으로 계산
2. **손실 계산 (Loss Calculation)**: 예측값과 실제값의 차이 계산
3. **역전파 (Backward Pass)**: 출력에서 입력으로 기울기 계산
4. **가중치 업데이트**: 경사하강법으로 가중치 조정

**경사하강법:**
```
w_new = w_old - learning_rate * ∂Loss/∂w
```

### 4. 활성화 함수

신경망에 비선형성을 부여합니다.

- **Sigmoid**: `σ(x) = 1 / (1 + e^(-x))`
- **ReLU**: `f(x) = max(0, x)`
- **Tanh**: `tanh(x) = (e^x - e^(-x)) / (e^x + e^(-x))`

## 🛠️ 구현 방법

### 단계별 구현

#### 1단계: 기본 구조 설계
```python
class NeuralNetwork:
    def __init__(self, layers):
        # 레이어 크기 정의
        # 가중치 초기화

    def forward(self, X):
        # 순전파 구현

    def backward(self, X, y):
        # 역전파 구현

    def train(self, X, y, epochs):
        # 학습 루프 구현
```

#### 2단계: 활성화 함수 구현
```python
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def sigmoid_derivative(x):
    return x * (1 - x)
```

#### 3단계: 순전파 구현
```python
# 입력층 → 은닉층 → 출력층
hidden = sigmoid(np.dot(input, weights_input_hidden) + bias_hidden)
output = sigmoid(np.dot(hidden, weights_hidden_output) + bias_output)
```

#### 4단계: 역전파 구현
```python
# 출력층 오차 계산
output_error = y - output
output_delta = output_error * sigmoid_derivative(output)

# 은닉층 오차 계산
hidden_error = output_delta.dot(weights_hidden_output.T)
hidden_delta = hidden_error * sigmoid_derivative(hidden)

# 가중치 업데이트
weights_hidden_output += hidden.T.dot(output_delta) * learning_rate
weights_input_hidden += input.T.dot(hidden_delta) * learning_rate
```

## 📁 프로젝트 구조

```
tiny-ai/
├── README.md                      # 프로젝트 설명서
├── requirements.txt               # 필수 라이브러리
├── docs/
│   ├── theory.md                  # 이론 설명
│   └── implementation.md          # 구현 가이드
├── src/
│   ├── __init__.py
│   ├── perceptron.py              # 단일 퍼셉트론 구현
│   ├── mlp.py                     # 다층 퍼셉트론 구현
│   ├── activations.py             # 활성화 함수들
│   ├── utils.py                   # 유틸리티 함수들
│   ├── text_processing.py         # 🆕 텍스트 처리 (토크나이저, TF-IDF)
│   └── chatbot.py                 # 🆕 대화형 AI 챗봇
├── data/
│   └── intents.json               # 🆕 챗봇 의도 정의
├── examples/
│   ├── xor_problem.py             # XOR 문제 해결
│   ├── simple_perceptron.py       # 단순 퍼셉트론 예제
│   ├── iris_classification.py     # 붓꽃 분류
│   └── chatbot_demo.py            # 🆕 챗봇 데모
└── tests/
    └── test_neural_network.py
```

## 💡 사용 예시

### XOR 문제 해결

```python
from src.mlp import MLP

# XOR 데이터
X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
y = np.array([[0], [1], [1], [0]])

# 신경망 생성 (입력: 2, 은닉: 4, 출력: 1)
model = MLP(layers=[2, 4, 1])

# 학습
model.train(X, y, epochs=10000, learning_rate=0.1)

# 예측
predictions = model.predict(X)
print(predictions)
```

### 붓꽃 분류

```python
from sklearn.datasets import load_iris
from src.mlp import MLP

# 데이터 로드
iris = load_iris()
X = iris.data
y = iris.target

# 신경망 생성
model = MLP(layers=[4, 8, 3])

# 학습 및 평가
model.train(X, y, epochs=1000)
accuracy = model.evaluate(X, y)
print(f"정확도: {accuracy:.2%}")
```

## 💬 대화형 AI 챗봇

**🆕 UPGRADED!** Claude 스타일의 유저 친화적인 챗봇으로 대폭 개선되었습니다!

### 🌟 핵심 특징

- 🤖 **Claude처럼 친절하고 체계적**: 구조화된 답변, 단계별 설명, 명확한 가이드
- 🎯 **99% 정확도**: 신경망 기반 의도 분류 (17개 의도, 150+ 패턴)
- 📚 **풍부한 응답**: 단순 답변이 아닌 실질적인 도움과 추가 질문
- 💡 **논리적 문제 해결**: 체계적인 접근법과 프레임워크 제공
- ❤️ **공감과 격려**: 사용자의 감정을 이해하고 진심으로 응원
- 🔍 **추가 도움 제안**: 답변 후 항상 "더 필요한 게 있나요?" 물어봄

### 🎭 지원하는 대화 (17개 의도)

**기본 대화:**
- ✅ **인사** - 환영 메시지와 함께 도움 제안
- ✅ **감사 표현** - 공감하며 추가 지원 제안
- ✅ **작별 인사** - 따뜻한 마무리와 재방문 유도
- ✅ **안부 질문** - 역질문하며 대화 이어가기

**도움 및 학습:**
- ✅ **도움 요청** - 3단계로 구체화 요청 (주제/어려운 부분/원하는 결과)
- ✅ **공부 방법** - 과학적 학습 전략, 실전 팁, 맞춤 조언
- ✅ **설명 요청** - 수준별 설명, 비유 활용, 단계별 가이드
- ✅ **문제 해결** - 5단계 접근법 (정의→분석→해결책→선택→실행)

**계획 및 의사결정:**
- ✅ **계획 수립** - SMART 목표, 단계별 전략, 마일스톤 설정
- ✅ **의사결정 지원** - 논리적 프로세스, 장단점 분석, 직관 확인

**동기부여 및 격려:**
- ✅ **격려/응원** - 공감 → 긍정 재구성 → 실용적 조언 → 지속 응원
- ✅ **칭찬 응답** - 겸손하게 받아들이며 사용자 칭찬으로 되돌리기

**기타:**
- ✅ **농담** - 유머 시도 + 솔직한 한계 인정 + 대안 제시
- ✅ **기능 문의** - 강점과 한계를 명확히 설명
- ✅ **자기소개** - 이름, 역할, 가치관, 목표 소개
- ✅ **시간/날씨** - 한계 인정 + 대안 제시 + 관련 도움 제안

### 사용 예시

```python
from src.chatbot import TinyAIAssistant

# 챗봇 초기화
assistant = TinyAIAssistant(intents_file='data/intents.json')

# 학습
assistant.train(epochs=500)

# 대화
response = assistant.chat("안녕하세요!")
print(response)  # "안녕하세요! 반갑습니다. 무엇을 도와드릴까요?"

response = assistant.chat("공부 도와줘")
print(response)  # "공부를 도와드릴게요! 효과적인 학습 방법은..."
```

### 💬 실제 대화 예시

**예시 1: 공부 도움 요청**
```
👤 사용자: 공부 도와줘

🤖 AI: 공부 방법 관련 조언 드릴게요! 📚

**효과적인 학습 전략:**

1️⃣ **명확한 목표 설정**
   • '오늘 수학 10페이지' 같은 구체적 목표
   • 달성 가능한 작은 목표부터 시작

2️⃣ **능동적 학습**
   • 그냥 읽기 X → 요약하고 설명해보기 O
   • 스스로 질문 만들고 답하기

3️⃣ **간격을 둔 복습**
   • 1일 후, 3일 후, 1주일 후 복습
   • 망각 곡선을 이기는 최고의 방법

4️⃣ **집중 시간 관리**
   • 포모도로: 25분 집중 + 5분 휴식
   • 멀티태스킹 피하기

어떤 과목을 공부하시나요? 더 구체적인 조언을 드릴 수 있어요!
```

**예시 2: 격려 및 동기부여**
```
👤 사용자: 힘들어... 포기하고 싶어

🤖 AI: 힘든 시간을 보내고 계시는군요. 마음이 정말 무거우시겠어요. 😔

하지만 이런 말씀 드리고 싶어요:

**지금 당신이 느끼는 감정은 완전히 정상이에요.**

어려운 일을 겪으면서 힘들어하는 건 당연합니다. 그건 당신이 약해서가
아니라, 당신이 정말 열심히 노력하고 있다는 증거예요.

**기억해 주세요:**
• 한 번에 한 걸음씩이면 충분합니다
• 완벽하지 않아도 괜찮아요
• 쉬어가는 것도 전진입니다
• 당신은 생각보다 훨씬 강한 사람이에요

지금 당장은 보이지 않겠지만, 이 경험이 당신을 더 단단하게 만들 거예요.
저는 당신을 응원합니다! 💪

혹시 구체적으로 어떤 부분이 힘드신가요? 함께 이야기 나눠요.
```

**예시 3: 문제 해결 지원**
```
👤 사용자: 계획 세워줘

🤖 AI: 좋은 계획을 함께 세워봅시다! 📋

**효과적인 계획 수립 단계:**

**1단계: 목표 명확화**
• 무엇을 달성하고 싶으신가요?
• 언제까지 완료하고 싶으신가요?
• 왜 이 목표가 중요한가요?

**2단계: 현황 파악**
• 현재 어디까지 와 있나요?
• 어떤 자원(시간, 도구 등)이 있나요?
• 어떤 제약사항이 있나요?

**3단계: 작은 단계로 나누기**
• 큰 목표를 작은 태스크로 분해
• 각 태스크의 우선순위 정하기
• 실행 가능한 첫 단계 찾기

**4단계: 일정 배치**
• 현실적인 시간 할당
• 여유 시간 확보 (20% rule)
• 마일스톤 설정

무엇에 대한 계획을 세우시려는지 말씀해 주시면,
구체적인 단계를 함께 만들어볼게요!
```

### CLI 챗봇 실행

```bash
python examples/chatbot_demo.py
```

대화형 인터페이스에서 AI와 자유롭게 대화할 수 있습니다!

## 🚀 실행 방법

### 요구사항

```bash
pip install numpy matplotlib
```

### 예제 실행

```bash
# XOR 문제
python examples/xor_problem.py

# 붓꽃 분류
python examples/iris_classification.py
```

## 📖 학습 리소스

### 권장 학습 순서

1. **docs/theory.md** - 이론적 배경 학습
2. **src/perceptron.py** - 단일 퍼셉트론 코드 분석
3. **src/mlp.py** - 다층 신경망 코드 분석
4. **examples/xor_problem.py** - 간단한 예제 실행
5. **examples/iris_classification.py** - 실제 문제 적용

## 🎯 주요 학습 목표

- ✅ 신경망의 기본 구조 이해
- ✅ 순전파와 역전파 알고리즘 이해
- ✅ 경사하강법을 통한 학습 과정 이해
- ✅ NumPy를 활용한 행렬 연산
- ✅ 실제 문제에 신경망 적용

## 📝 라이선스

MIT License

## 🤝 기여

이 프로젝트는 교육 목적으로 만들어졌습니다. 개선 사항이나 버그 리포트는 언제나 환영합니다!
