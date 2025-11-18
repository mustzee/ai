# Tiny AI - 초경량 인공지능 프로젝트

Tiny AI는 최소한의 코드로 인공지능의 핵심 원리를 이해하고 구현하는 교육용 프로젝트입니다.

## 📚 목차

1. [Tiny AI란?](#tiny-ai란)
2. [핵심 원리](#핵심-원리)
3. [구현 방법](#구현-방법)
4. [프로젝트 구조](#프로젝트-구조)
5. [사용 예시](#사용-예시)
6. [실행 방법](#실행-방법)

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
├── README.md                 # 프로젝트 설명서
├── docs/
│   ├── theory.md            # 이론 설명
│   └── implementation.md    # 구현 가이드
├── src/
│   ├── __init__.py
│   ├── perceptron.py        # 단일 퍼셉트론 구현
│   ├── mlp.py               # 다층 퍼셉트론 구현
│   ├── activations.py       # 활성화 함수들
│   └── utils.py             # 유틸리티 함수들
├── examples/
│   ├── xor_problem.py       # XOR 문제 해결
│   ├── mnist_simple.py      # 간단한 숫자 인식
│   └── iris_classification.py # 붓꽃 분류
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
