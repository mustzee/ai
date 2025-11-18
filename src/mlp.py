"""
다층 퍼셉트론 (Multi-Layer Perceptron, MLP) 구현

역전파 알고리즘을 사용하여 학습하는 완전 연결 신경망입니다.
"""

import numpy as np
from .activations import get_activation, sigmoid, sigmoid_derivative


class MLP:
    """
    다층 퍼셉트론 클래스

    여러 층의 뉴런으로 구성된 완전 연결 신경망입니다.
    역전파 알고리즘을 통해 학습합니다.

    Attributes:
        layers: 각 층의 뉴런 개수 리스트
        weights: 각 층의 가중치 행렬 리스트
        biases: 각 층의 편향 벡터 리스트
        activation: 활성화 함수 이름
    """

    def __init__(self, layers, activation='sigmoid', learning_rate=0.1):
        """
        MLP 초기화

        Args:
            layers: 각 층의 뉴런 개수 리스트
                   예: [2, 4, 1] = 입력 2, 은닉층 4, 출력 1
            activation: 활성화 함수 ('sigmoid', 'relu', 'tanh')
            learning_rate: 학습률
        """
        self.layers = layers
        self.num_layers = len(layers)
        self.learning_rate = learning_rate
        self.activation_name = activation

        # 활성화 함수 설정
        self.activation_func, self.activation_derivative = get_activation(activation)

        # 가중치와 편향 초기화
        self.weights = []
        self.biases = []

        # He 초기화 (ReLU에 적합) 또는 Xavier 초기화 (Sigmoid/Tanh에 적합)
        for i in range(len(layers) - 1):
            # Xavier/Glorot 초기화
            limit = np.sqrt(6 / (layers[i] + layers[i + 1]))
            w = np.random.uniform(-limit, limit, (layers[i], layers[i + 1]))
            b = np.zeros((1, layers[i + 1]))

            self.weights.append(w)
            self.biases.append(b)

        # 학습 히스토리
        self.history = {
            'loss': [],
            'accuracy': []
        }

    def forward(self, X):
        """
        순전파 (Forward Propagation)

        입력에서 출력으로 데이터를 전달하며 각 층의 활성화값을 계산합니다.

        Args:
            X: 입력 데이터 (n_samples, n_features)

        Returns:
            각 층의 활성화값 리스트
        """
        activations = [X]

        for i in range(len(self.weights)):
            # 선형 변환: z = w·a + b
            z = np.dot(activations[-1], self.weights[i]) + self.biases[i]

            # 활성화 함수 적용
            a = self.activation_func(z)
            activations.append(a)

        return activations

    def backward(self, X, y, activations):
        """
        역전파 (Backward Propagation)

        출력층에서 입력층으로 오차를 전파하며 기울기를 계산합니다.

        Args:
            X: 입력 데이터
            y: 타겟 데이터
            activations: 순전파에서 계산된 활성화값들

        Returns:
            각 층의 가중치와 편향에 대한 기울기
        """
        m = X.shape[0]  # 샘플 개수
        gradients_w = []
        gradients_b = []

        # 출력층 오차 계산
        # δL = (a - y) ⊙ σ'(z)
        delta = (activations[-1] - y) * self.activation_derivative(activations[-1])

        # 역방향으로 기울기 계산
        for i in range(len(self.weights) - 1, -1, -1):
            # 가중치 기울기: ∂L/∂w = aᵀ · δ
            grad_w = np.dot(activations[i].T, delta) / m

            # 편향 기울기: ∂L/∂b = mean(δ)
            grad_b = np.mean(delta, axis=0, keepdims=True)

            gradients_w.insert(0, grad_w)
            gradients_b.insert(0, grad_b)

            # 이전 층으로 오차 전파 (출력층이 아닌 경우)
            if i > 0:
                # δˡ⁻¹ = (δˡ · wˡᵀ) ⊙ σ'(zˡ⁻¹)
                delta = np.dot(delta, self.weights[i].T) * \
                        self.activation_derivative(activations[i])

        return gradients_w, gradients_b

    def update_weights(self, gradients_w, gradients_b):
        """
        경사하강법으로 가중치 업데이트

        w = w - learning_rate * ∂L/∂w

        Args:
            gradients_w: 가중치 기울기
            gradients_b: 편향 기울기
        """
        for i in range(len(self.weights)):
            self.weights[i] -= self.learning_rate * gradients_w[i]
            self.biases[i] -= self.learning_rate * gradients_b[i]

    def train(self, X, y, epochs=1000, batch_size=None, verbose=True):
        """
        신경망 학습

        Args:
            X: 학습 데이터 (n_samples, n_features)
            y: 타겟 데이터 (n_samples, n_outputs)
            epochs: 학습 반복 횟수
            batch_size: 미니배치 크기 (None이면 전체 배치)
            verbose: 학습 과정 출력 여부

        Returns:
            학습 히스토리
        """
        # y가 1차원이면 2차원으로 변환
        if y.ndim == 1:
            y = y.reshape(-1, 1)

        n_samples = X.shape[0]

        for epoch in range(epochs):
            # 미니배치 학습
            if batch_size is None:
                batch_size = n_samples

            # 데이터 섞기
            indices = np.random.permutation(n_samples)
            X_shuffled = X[indices]
            y_shuffled = y[indices]

            # 배치별 학습
            for start_idx in range(0, n_samples, batch_size):
                end_idx = min(start_idx + batch_size, n_samples)
                X_batch = X_shuffled[start_idx:end_idx]
                y_batch = y_shuffled[start_idx:end_idx]

                # 순전파
                activations = self.forward(X_batch)

                # 역전파
                gradients_w, gradients_b = self.backward(X_batch, y_batch, activations)

                # 가중치 업데이트
                self.update_weights(gradients_w, gradients_b)

            # 전체 데이터에 대한 손실 계산
            predictions = self.predict(X)
            loss = self.compute_loss(y, predictions)
            self.history['loss'].append(loss)

            # 정확도 계산 (분류 문제인 경우)
            if self.layers[-1] == 1:  # 이진 분류
                accuracy = np.mean((predictions > 0.5) == y)
            else:  # 다중 분류
                accuracy = np.mean(np.argmax(predictions, axis=1) == np.argmax(y, axis=1))
            self.history['accuracy'].append(accuracy)

            # 진행상황 출력
            if verbose and ((epoch + 1) % (epochs // 10) == 0 or epoch == 0):
                print(f"Epoch {epoch + 1}/{epochs}, Loss: {loss:.4f}, Accuracy: {accuracy:.4f}")

        return self.history

    def predict(self, X):
        """
        예측 수행

        Args:
            X: 입력 데이터

        Returns:
            예측값
        """
        activations = self.forward(X)
        return activations[-1]

    def compute_loss(self, y_true, y_pred):
        """
        손실 함수 계산 (MSE)

        L = (1/n) * Σ(y - ŷ)²

        Args:
            y_true: 실제값
            y_pred: 예측값

        Returns:
            평균 제곱 오차
        """
        return np.mean((y_true - y_pred) ** 2)

    def evaluate(self, X, y):
        """
        모델 평가

        Args:
            X: 테스트 데이터
            y: 실제 레이블

        Returns:
            정확도
        """
        predictions = self.predict(X)

        if y.ndim == 1:
            y = y.reshape(-1, 1)

        if self.layers[-1] == 1:  # 이진 분류
            accuracy = np.mean((predictions > 0.5) == y)
        else:  # 다중 분류
            accuracy = np.mean(np.argmax(predictions, axis=1) == np.argmax(y, axis=1))

        return accuracy

    def save_weights(self, filepath):
        """
        가중치 저장

        Args:
            filepath: 저장 경로
        """
        np.savez(filepath, weights=self.weights, biases=self.biases)
        print(f"Weights saved to {filepath}")

    def load_weights(self, filepath):
        """
        가중치 로드

        Args:
            filepath: 로드 경로
        """
        data = np.load(filepath, allow_pickle=True)
        self.weights = list(data['weights'])
        self.biases = list(data['biases'])
        print(f"Weights loaded from {filepath}")
