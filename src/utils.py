"""
유틸리티 함수 모듈

데이터 전처리, 시각화 등 보조 기능을 제공합니다.
"""

import numpy as np


def normalize(X, axis=0):
    """
    데이터 정규화 (Min-Max Scaling)

    각 특성을 0과 1 사이로 스케일링합니다.

    수식: x_norm = (x - min) / (max - min)

    Args:
        X: 입력 데이터
        axis: 정규화할 축 (0: 열별, 1: 행별)

    Returns:
        정규화된 데이터
    """
    min_val = np.min(X, axis=axis, keepdims=True)
    max_val = np.max(X, axis=axis, keepdims=True)

    # 0으로 나누는 것 방지
    range_val = max_val - min_val
    range_val[range_val == 0] = 1

    return (X - min_val) / range_val


def standardize(X, axis=0):
    """
    데이터 표준화 (Z-score Normalization)

    각 특성을 평균 0, 표준편차 1로 변환합니다.

    수식: x_std = (x - μ) / σ

    Args:
        X: 입력 데이터
        axis: 표준화할 축

    Returns:
        표준화된 데이터
    """
    mean = np.mean(X, axis=axis, keepdims=True)
    std = np.std(X, axis=axis, keepdims=True)

    # 0으로 나누는 것 방지
    std[std == 0] = 1

    return (X - mean) / std


def one_hot_encode(y, num_classes=None):
    """
    원-핫 인코딩

    정수 레이블을 원-핫 벡터로 변환합니다.

    예: [0, 1, 2] -> [[1,0,0], [0,1,0], [0,0,1]]

    Args:
        y: 정수 레이블 배열
        num_classes: 클래스 개수 (None이면 자동 계산)

    Returns:
        원-핫 인코딩된 배열
    """
    if num_classes is None:
        num_classes = int(np.max(y)) + 1

    one_hot = np.zeros((len(y), num_classes))
    one_hot[np.arange(len(y)), y.astype(int)] = 1

    return one_hot


def train_test_split(X, y, test_size=0.2, random_state=None):
    """
    데이터를 학습/테스트 세트로 분할

    Args:
        X: 입력 데이터
        y: 타겟 데이터
        test_size: 테스트 세트 비율 (0~1)
        random_state: 난수 시드

    Returns:
        X_train, X_test, y_train, y_test
    """
    if random_state is not None:
        np.random.seed(random_state)

    # 인덱스 섞기
    n_samples = len(X)
    indices = np.random.permutation(n_samples)

    # 분할 지점 계산
    test_size = int(n_samples * test_size)
    train_indices = indices[test_size:]
    test_indices = indices[:test_size]

    return X[train_indices], X[test_indices], y[train_indices], y[test_indices]


def compute_accuracy(y_true, y_pred, threshold=0.5):
    """
    정확도 계산

    Args:
        y_true: 실제 레이블
        y_pred: 예측값
        threshold: 이진 분류 임계값

    Returns:
        정확도 (0~1)
    """
    if y_pred.shape[1] == 1:  # 이진 분류
        predictions = (y_pred > threshold).astype(int)
    else:  # 다중 분류
        predictions = np.argmax(y_pred, axis=1)
        y_true = np.argmax(y_true, axis=1) if y_true.ndim > 1 else y_true

    return np.mean(predictions == y_true)


def create_batches(X, y, batch_size):
    """
    미니배치 생성기

    Args:
        X: 입력 데이터
        y: 타겟 데이터
        batch_size: 배치 크기

    Yields:
        (X_batch, y_batch) 튜플
    """
    n_samples = len(X)
    indices = np.random.permutation(n_samples)

    for start_idx in range(0, n_samples, batch_size):
        end_idx = min(start_idx + batch_size, n_samples)
        batch_indices = indices[start_idx:end_idx]
        yield X[batch_indices], y[batch_indices]


def plot_decision_boundary(model, X, y, resolution=0.02):
    """
    결정 경계 시각화 (2D 데이터)

    Args:
        model: 학습된 모델
        X: 입력 데이터 (2D)
        y: 레이블
        resolution: 그리드 해상도
    """
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib이 설치되지 않았습니다.")
        return

    # 특성 범위 설정
    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1

    # 그리드 생성
    xx, yy = np.meshgrid(np.arange(x_min, x_max, resolution),
                         np.arange(y_min, y_max, resolution))

    # 각 그리드 포인트에 대한 예측
    Z = model.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)

    # 결정 경계 그리기
    plt.contourf(xx, yy, Z, alpha=0.4)
    plt.scatter(X[:, 0], X[:, 1], c=y, alpha=0.8, edgecolors='k')
    plt.xlabel('Feature 1')
    plt.ylabel('Feature 2')
    plt.title('Decision Boundary')
    plt.show()


def plot_learning_curve(history):
    """
    학습 곡선 시각화

    Args:
        history: 학습 히스토리 딕셔너리
    """
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib이 설치되지 않았습니다.")
        return

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))

    # 손실 곡선
    ax1.plot(history['loss'])
    ax1.set_title('Model Loss')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.grid(True)

    # 정확도 곡선
    ax2.plot(history['accuracy'])
    ax2.set_title('Model Accuracy')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Accuracy')
    ax2.grid(True)

    plt.tight_layout()
    plt.show()


class EarlyStopping:
    """
    조기 종료 콜백

    검증 손실이 개선되지 않으면 학습을 조기 종료합니다.
    """

    def __init__(self, patience=10, min_delta=0.0001):
        """
        Args:
            patience: 개선되지 않아도 기다릴 에포크 수
            min_delta: 개선으로 간주할 최소 변화량
        """
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_loss = None
        self.early_stop = False

    def __call__(self, val_loss):
        """
        검증 손실 확인

        Args:
            val_loss: 현재 검증 손실

        Returns:
            조기 종료 여부
        """
        if self.best_loss is None:
            self.best_loss = val_loss
        elif val_loss > self.best_loss - self.min_delta:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_loss = val_loss
            self.counter = 0

        return self.early_stop
