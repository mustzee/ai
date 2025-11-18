"""
활성화 함수 모듈

신경망에서 사용되는 다양한 활성화 함수와 그 도함수를 제공합니다.
"""

import numpy as np


def sigmoid(x):
    """
    시그모이드 활성화 함수

    수식: σ(x) = 1 / (1 + e^(-x))
    범위: (0, 1)
    용도: 이진 분류, 확률 출력

    Args:
        x: 입력 값 (numpy array)

    Returns:
        시그모이드 함수 적용 결과
    """
    # 오버플로우 방지
    x = np.clip(x, -500, 500)
    return 1 / (1 + np.exp(-x))


def sigmoid_derivative(x):
    """
    시그모이드 함수의 도함수

    수식: σ'(x) = σ(x) * (1 - σ(x))

    Args:
        x: 시그모이드 함수의 출력값

    Returns:
        도함수 값
    """
    return x * (1 - x)


def relu(x):
    """
    ReLU (Rectified Linear Unit) 활성화 함수

    수식: f(x) = max(0, x)
    범위: [0, ∞)
    용도: 은닉층에서 널리 사용, 기울기 소실 문제 완화

    Args:
        x: 입력 값

    Returns:
        ReLU 함수 적용 결과
    """
    return np.maximum(0, x)


def relu_derivative(x):
    """
    ReLU 함수의 도함수

    수식: f'(x) = 1 if x > 0 else 0

    Args:
        x: 입력 값

    Returns:
        도함수 값
    """
    return (x > 0).astype(float)


def tanh(x):
    """
    Tanh (Hyperbolic Tangent) 활성화 함수

    수식: tanh(x) = (e^x - e^(-x)) / (e^x + e^(-x))
    범위: (-1, 1)
    용도: 시그모이드보다 강한 기울기

    Args:
        x: 입력 값

    Returns:
        Tanh 함수 적용 결과
    """
    return np.tanh(x)


def tanh_derivative(x):
    """
    Tanh 함수의 도함수

    수식: tanh'(x) = 1 - tanh²(x)

    Args:
        x: Tanh 함수의 출력값

    Returns:
        도함수 값
    """
    return 1 - x ** 2


def softmax(x):
    """
    Softmax 활성화 함수

    다중 클래스 분류에서 사용되며, 출력을 확률 분포로 변환합니다.

    수식: softmax(xi) = e^xi / Σ(e^xj)

    Args:
        x: 입력 값 (numpy array)

    Returns:
        확률 분포 (합이 1)
    """
    # 수치 안정성을 위해 최대값을 빼줌
    exp_x = np.exp(x - np.max(x, axis=-1, keepdims=True))
    return exp_x / np.sum(exp_x, axis=-1, keepdims=True)


def linear(x):
    """
    선형 활성화 함수 (항등 함수)

    수식: f(x) = x
    용도: 회귀 문제의 출력층

    Args:
        x: 입력 값

    Returns:
        입력값 그대로
    """
    return x


def linear_derivative(x):
    """
    선형 함수의 도함수

    수식: f'(x) = 1

    Args:
        x: 입력 값

    Returns:
        1
    """
    return np.ones_like(x)


# 활성화 함수 매핑 딕셔너리
ACTIVATION_FUNCTIONS = {
    'sigmoid': (sigmoid, sigmoid_derivative),
    'relu': (relu, relu_derivative),
    'tanh': (tanh, tanh_derivative),
    'softmax': (softmax, None),  # Softmax는 보통 크로스엔트로피와 함께 사용
    'linear': (linear, linear_derivative),
}


def get_activation(name):
    """
    활성화 함수 이름으로 함수와 도함수를 반환

    Args:
        name: 활성화 함수 이름

    Returns:
        (활성화 함수, 도함수) 튜플
    """
    if name not in ACTIVATION_FUNCTIONS:
        raise ValueError(f"Unknown activation function: {name}. "
                        f"Available: {list(ACTIVATION_FUNCTIONS.keys())}")
    return ACTIVATION_FUNCTIONS[name]
