"""
단일 퍼셉트론 구현

가장 기본적인 인공 신경망 단위인 퍼셉트론을 구현합니다.
"""

import numpy as np
from .activations import sigmoid, sigmoid_derivative


class Perceptron:
    """
    단일 퍼셉트론 클래스

    선형 분류 문제를 해결할 수 있는 가장 기본적인 신경망입니다.

    Attributes:
        weights: 가중치 벡터
        bias: 편향 값
        learning_rate: 학습률
    """

    def __init__(self, input_size, learning_rate=0.1):
        """
        퍼셉트론 초기화

        Args:
            input_size: 입력 특성의 개수
            learning_rate: 학습률 (기본값: 0.1)
        """
        # 가중치를 작은 랜덤 값으로 초기화 (Xavier 초기화)
        self.weights = np.random.randn(input_size) * np.sqrt(2.0 / input_size)
        self.bias = 0.0
        self.learning_rate = learning_rate

    def activate(self, x):
        """
        활성화 함수 (계단 함수)

        Args:
            x: 입력 값

        Returns:
            0 또는 1
        """
        return 1 if x >= 0 else 0

    def predict(self, X):
        """
        예측 수행

        수식: y = activate(w·x + b)

        Args:
            X: 입력 데이터 (n_samples, n_features)

        Returns:
            예측값 (n_samples,)
        """
        # X가 1차원이면 2차원으로 변환
        if X.ndim == 1:
            X = X.reshape(1, -1)

        # 선형 결합 계산
        linear_output = np.dot(X, self.weights) + self.bias

        # 활성화 함수 적용
        return np.array([self.activate(x) for x in linear_output])

    def train(self, X, y, epochs=100):
        """
        퍼셉트론 학습

        퍼셉트론 학습 규칙:
        w_new = w_old + learning_rate * (y_true - y_pred) * x

        Args:
            X: 학습 데이터 (n_samples, n_features)
            y: 타겟 데이터 (n_samples,)
            epochs: 학습 반복 횟수

        Returns:
            각 에포크별 손실 리스트
        """
        losses = []

        for epoch in range(epochs):
            total_error = 0

            for xi, yi in zip(X, y):
                # 예측
                prediction = self.predict(xi)[0]

                # 오차 계산
                error = yi - prediction

                # 가중치 업데이트
                self.weights += self.learning_rate * error * xi
                self.bias += self.learning_rate * error

                total_error += abs(error)

            # 평균 오차 기록
            avg_error = total_error / len(X)
            losses.append(avg_error)

            # 10 에포크마다 진행상황 출력
            if (epoch + 1) % 10 == 0 or epoch == 0:
                print(f"Epoch {epoch + 1}/{epochs}, Error: {avg_error:.4f}")

            # 완벽하게 학습되면 조기 종료
            if avg_error == 0:
                print(f"Perfect classification achieved at epoch {epoch + 1}")
                break

        return losses

    def get_weights(self):
        """
        현재 가중치와 편향 반환

        Returns:
            (weights, bias) 튜플
        """
        return self.weights.copy(), self.bias


class SmoothPerceptron(Perceptron):
    """
    부드러운 활성화 함수를 사용하는 퍼셉트론

    계단 함수 대신 시그모이드 함수를 사용하여
    더 부드러운 학습이 가능합니다.
    """

    def activate(self, x):
        """
        시그모이드 활성화 함수

        Args:
            x: 입력 값

        Returns:
            0과 1 사이의 값
        """
        return sigmoid(x)

    def predict(self, X):
        """
        예측 수행 (확률 출력)

        Args:
            X: 입력 데이터

        Returns:
            0과 1 사이의 확률값
        """
        if X.ndim == 1:
            X = X.reshape(1, -1)

        linear_output = np.dot(X, self.weights) + self.bias
        return sigmoid(linear_output)

    def train(self, X, y, epochs=100):
        """
        경사하강법을 사용한 학습

        손실 함수: MSE (Mean Squared Error)
        L = (1/n) * Σ(yi - ŷi)²

        Args:
            X: 학습 데이터
            y: 타겟 데이터
            epochs: 학습 반복 횟수

        Returns:
            각 에포크별 손실 리스트
        """
        losses = []

        for epoch in range(epochs):
            # 순전파
            predictions = self.predict(X)

            # 손실 계산 (MSE)
            loss = np.mean((y - predictions) ** 2)
            losses.append(loss)

            # 기울기 계산
            error = y - predictions
            d_weights = -2 * np.dot(X.T, error) / len(X)
            d_bias = -2 * np.mean(error)

            # 가중치 업데이트
            self.weights -= self.learning_rate * d_weights
            self.bias -= self.learning_rate * d_bias

            # 진행상황 출력
            if (epoch + 1) % 100 == 0 or epoch == 0:
                print(f"Epoch {epoch + 1}/{epochs}, Loss: {loss:.4f}")

        return losses
