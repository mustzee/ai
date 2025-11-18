# Tiny AI 구현 가이드

처음부터 신경망을 만드는 단계별 가이드입니다.

## 목차

1. [환경 설정](#환경-설정)
2. [단계별 구현](#단계별-구현)
3. [완전한 예제](#완전한-예제)
4. [디버깅 팁](#디버깅-팁)
5. [성능 최적화](#성능-최적화)

---

## 환경 설정

### 필수 라이브러리 설치

```bash
pip install numpy matplotlib
```

### 선택적 라이브러리

```bash
pip install scikit-learn  # 데이터셋 및 전처리
```

---

## 단계별 구현

### 1단계: 활성화 함수 구현

가장 기본적인 구성 요소부터 시작합니다.

```python
import numpy as np

def sigmoid(x):
    """시그모이드 함수"""
    return 1 / (1 + np.exp(-x))

def sigmoid_derivative(x):
    """시그모이드 도함수"""
    return x * (1 - x)
```

**테스트:**
```python
# 입력값
x = np.array([0, 1, 2, -1])

# 출력값
y = sigmoid(x)
print(y)  # [0.5, 0.73, 0.88, 0.27]

# 도함수
dy = sigmoid_derivative(y)
print(dy)  # [0.25, 0.20, 0.10, 0.20]
```

---

### 2단계: 가중치 초기화

```python
def initialize_weights(layers):
    """
    가중치와 편향 초기화

    Args:
        layers: 각 층의 뉴런 개수 리스트 [2, 4, 1]

    Returns:
        weights: 가중치 리스트
        biases: 편향 리스트
    """
    weights = []
    biases = []

    for i in range(len(layers) - 1):
        # Xavier 초기화
        limit = np.sqrt(6 / (layers[i] + layers[i+1]))
        w = np.random.uniform(-limit, limit,
                             (layers[i], layers[i+1]))
        b = np.zeros((1, layers[i+1]))

        weights.append(w)
        biases.append(b)

    return weights, biases
```

**테스트:**
```python
layers = [2, 4, 1]  # 입력 2, 은닉 4, 출력 1
weights, biases = initialize_weights(layers)

print(f"W1 shape: {weights[0].shape}")  # (2, 4)
print(f"W2 shape: {weights[1].shape}")  # (4, 1)
print(f"b1 shape: {biases[0].shape}")   # (1, 4)
print(f"b2 shape: {biases[1].shape}")   # (1, 1)
```

---

### 3단계: 순전파 구현

```python
def forward_propagation(X, weights, biases):
    """
    순전파 수행

    Args:
        X: 입력 데이터 (n_samples, n_features)
        weights: 가중치 리스트
        biases: 편향 리스트

    Returns:
        activations: 각 층의 활성화값 리스트
    """
    activations = [X]

    # 각 층을 통과
    for w, b in zip(weights, biases):
        # 선형 변환
        z = np.dot(activations[-1], w) + b

        # 활성화 함수 적용
        a = sigmoid(z)

        activations.append(a)

    return activations
```

**테스트:**
```python
# 샘플 데이터
X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])

# 순전파
activations = forward_propagation(X, weights, biases)

print(f"입력 shape: {activations[0].shape}")    # (4, 2)
print(f"은닉층 shape: {activations[1].shape}")  # (4, 4)
print(f"출력 shape: {activations[2].shape}")    # (4, 1)
```

---

### 4단계: 역전파 구현

```python
def backward_propagation(X, y, activations, weights):
    """
    역전파 수행

    Args:
        X: 입력 데이터
        y: 타겟 데이터
        activations: 순전파에서 계산된 활성화값
        weights: 현재 가중치

    Returns:
        gradients_w: 가중치 기울기
        gradients_b: 편향 기울기
    """
    m = X.shape[0]  # 샘플 수
    gradients_w = []
    gradients_b = []

    # 출력층 오차
    delta = (activations[-1] - y) * sigmoid_derivative(activations[-1])

    # 역방향으로 기울기 계산
    for i in range(len(weights) - 1, -1, -1):
        # 기울기 계산
        grad_w = np.dot(activations[i].T, delta) / m
        grad_b = np.mean(delta, axis=0, keepdims=True)

        gradients_w.insert(0, grad_w)
        gradients_b.insert(0, grad_b)

        # 이전 층으로 오차 전파
        if i > 0:
            delta = np.dot(delta, weights[i].T) * \
                   sigmoid_derivative(activations[i])

    return gradients_w, gradients_b
```

---

### 5단계: 가중치 업데이트

```python
def update_weights(weights, biases, gradients_w, gradients_b,
                  learning_rate=0.1):
    """
    경사하강법으로 가중치 업데이트

    Args:
        weights: 현재 가중치
        biases: 현재 편향
        gradients_w: 가중치 기울기
        gradients_b: 편향 기울기
        learning_rate: 학습률
    """
    for i in range(len(weights)):
        weights[i] -= learning_rate * gradients_w[i]
        biases[i] -= learning_rate * gradients_b[i]
```

---

### 6단계: 학습 루프

```python
def train(X, y, layers, epochs=1000, learning_rate=0.1):
    """
    신경망 학습

    Args:
        X: 학습 데이터
        y: 타겟 데이터
        layers: 층 구조
        epochs: 학습 반복 횟수
        learning_rate: 학습률

    Returns:
        weights, biases, history
    """
    # 초기화
    weights, biases = initialize_weights(layers)
    history = {'loss': [], 'accuracy': []}

    for epoch in range(epochs):
        # 순전파
        activations = forward_propagation(X, weights, biases)

        # 역전파
        gradients_w, gradients_b = backward_propagation(
            X, y, activations, weights
        )

        # 가중치 업데이트
        update_weights(weights, biases, gradients_w,
                      gradients_b, learning_rate)

        # 손실 계산
        predictions = activations[-1]
        loss = np.mean((y - predictions) ** 2)
        history['loss'].append(loss)

        # 정확도 계산
        accuracy = np.mean((predictions > 0.5) == y)
        history['accuracy'].append(accuracy)

        # 진행상황 출력
        if (epoch + 1) % 100 == 0:
            print(f"Epoch {epoch+1}/{epochs}, "
                  f"Loss: {loss:.4f}, "
                  f"Accuracy: {accuracy:.4f}")

    return weights, biases, history
```

---

## 완전한 예제

### XOR 문제 해결

```python
import numpy as np

# 1. 데이터 준비
X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
y = np.array([[0], [1], [1], [0]])

# 2. 신경망 구조 정의
layers = [2, 4, 1]  # 입력 2, 은닉 4, 출력 1

# 3. 학습
weights, biases, history = train(
    X, y, layers,
    epochs=5000,
    learning_rate=0.5
)

# 4. 예측
activations = forward_propagation(X, weights, biases)
predictions = activations[-1]

# 5. 결과 출력
print("\n예측 결과:")
for i in range(len(X)):
    pred = predictions[i][0]
    actual = y[i][0]
    print(f"{X[i]} → {pred:.4f} (실제: {actual})")

# 6. 시각화
import matplotlib.pyplot as plt

plt.figure(figsize=(10, 4))

plt.subplot(1, 2, 1)
plt.plot(history['loss'])
plt.title('Loss')
plt.xlabel('Epoch')
plt.ylabel('MSE')

plt.subplot(1, 2, 2)
plt.plot(history['accuracy'])
plt.title('Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')

plt.tight_layout()
plt.show()
```

---

## 디버깅 팁

### 1. 기울기 확인 (Gradient Checking)

수치적 기울기와 역전파 기울기를 비교합니다.

```python
def numerical_gradient(X, y, weights, biases, epsilon=1e-7):
    """수치적 기울기 계산"""
    numerical_grads = []

    for w in weights:
        grad = np.zeros_like(w)

        it = np.nditer(w, flags=['multi_index'])
        while not it.finished:
            idx = it.multi_index

            # w[idx] + epsilon
            old_value = w[idx]
            w[idx] = old_value + epsilon
            activations_plus = forward_propagation(X, weights, biases)
            loss_plus = np.mean((y - activations_plus[-1]) ** 2)

            # w[idx] - epsilon
            w[idx] = old_value - epsilon
            activations_minus = forward_propagation(X, weights, biases)
            loss_minus = np.mean((y - activations_minus[-1]) ** 2)

            # 기울기 계산
            grad[idx] = (loss_plus - loss_minus) / (2 * epsilon)

            # 원래 값으로 복원
            w[idx] = old_value
            it.iternext()

        numerical_grads.append(grad)

    return numerical_grads

# 사용법
activations = forward_propagation(X, weights, biases)
analytical_grads, _ = backward_propagation(X, y, activations, weights)
numerical_grads = numerical_gradient(X, y, weights, biases)

# 비교
for i, (ag, ng) in enumerate(zip(analytical_grads, numerical_grads)):
    diff = np.linalg.norm(ag - ng) / (np.linalg.norm(ag) + np.linalg.norm(ng))
    print(f"Layer {i} gradient difference: {diff:.10f}")
    # diff < 1e-7 이면 정상
```

### 2. 일반적인 문제와 해결책

#### 문제 1: 손실이 감소하지 않음

**원인:**
- 학습률이 너무 작음
- 가중치 초기화 문제
- 데이터 정규화 안 됨

**해결:**
```python
# 학습률 조정
learning_rate = 0.01  # → 0.1로 증가

# 데이터 정규화
X = (X - X.mean()) / X.std()

# He 초기화 시도
w = np.random.randn(n_in, n_out) * np.sqrt(2.0 / n_in)
```

#### 문제 2: 손실이 발산함

**원인:**
- 학습률이 너무 큼
- 기울기 폭발

**해결:**
```python
# 학습률 감소
learning_rate = 0.001

# 기울기 클리핑
max_grad = 5.0
for grad in gradients_w:
    np.clip(grad, -max_grad, max_grad, out=grad)
```

#### 문제 3: 과적합

**원인:**
- 모델이 너무 복잡
- 학습 데이터가 부족

**해결:**
```python
# L2 정규화 추가
lambda_reg = 0.01
for w in weights:
    w -= learning_rate * lambda_reg * w

# 드롭아웃
dropout_rate = 0.5
mask = np.random.rand(*hidden.shape) > dropout_rate
hidden *= mask
```

### 3. 학습 모니터링

```python
def monitor_training(history, window=100):
    """학습 과정 모니터링"""

    # 최근 window 개의 손실 평균
    recent_loss = np.mean(history['loss'][-window:])

    # 손실 변화율
    if len(history['loss']) > window:
        old_loss = np.mean(history['loss'][-2*window:-window])
        improvement = (old_loss - recent_loss) / old_loss * 100
        print(f"최근 {window} epoch 개선율: {improvement:.2f}%")

    # 조기 종료 체크
    patience = 200
    if len(history['loss']) > patience:
        recent = history['loss'][-patience:]
        if recent[-1] >= min(recent):
            print("경고: 손실이 개선되지 않고 있습니다!")
```

---

## 성능 최적화

### 1. 벡터화

**나쁜 예 (반복문):**
```python
output = np.zeros((m, n))
for i in range(m):
    for j in range(n):
        output[i, j] = sigmoid(np.dot(X[i], W[:, j]))
```

**좋은 예 (벡터화):**
```python
output = sigmoid(np.dot(X, W))
```

### 2. 미니배치 처리

```python
def create_mini_batches(X, y, batch_size=32):
    """미니배치 생성"""
    m = X.shape[0]
    mini_batches = []

    # 데이터 섞기
    permutation = np.random.permutation(m)
    X_shuffled = X[permutation]
    y_shuffled = y[permutation]

    # 배치 생성
    num_complete_batches = m // batch_size

    for k in range(num_complete_batches):
        X_batch = X_shuffled[k*batch_size:(k+1)*batch_size]
        y_batch = y_shuffled[k*batch_size:(k+1)*batch_size]
        mini_batches.append((X_batch, y_batch))

    # 마지막 배치
    if m % batch_size != 0:
        X_batch = X_shuffled[num_complete_batches*batch_size:]
        y_batch = y_shuffled[num_complete_batches*batch_size:]
        mini_batches.append((X_batch, y_batch))

    return mini_batches
```

### 3. 학습률 스케줄링

```python
def learning_rate_decay(initial_lr, epoch, decay_rate=0.95):
    """학습률 감소"""
    return initial_lr * (decay_rate ** epoch)

# 사용
for epoch in range(epochs):
    lr = learning_rate_decay(initial_lr=0.1, epoch=epoch)
    update_weights(weights, biases, gradients_w, gradients_b, lr)
```

---

## 다음 단계

### 추가 기능 구현

1. **다양한 활성화 함수**
   - ReLU, Leaky ReLU
   - Tanh
   - Softmax

2. **최적화 알고리즘**
   - Momentum
   - Adam
   - RMSprop

3. **정규화 기법**
   - L1/L2 정규화
   - 드롭아웃
   - 배치 정규화

4. **고급 기능**
   - 모델 저장/로드
   - 조기 종료
   - 학습률 스케줄링

---

## 참고 코드

전체 구현은 `src/` 디렉토리의 다음 파일들을 참고하세요:

- `src/activations.py`: 활성화 함수
- `src/perceptron.py`: 단일 퍼셉트론
- `src/mlp.py`: 다층 퍼셉트론
- `src/utils.py`: 유틸리티 함수

예제는 `examples/` 디렉토리에서 확인할 수 있습니다.
