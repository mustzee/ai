"""
붓꽃 분류 예제

Iris 데이터셋을 사용한 다중 클래스 분류 문제입니다.
3종류의 붓꽃을 4개의 특성으로 분류합니다.

데이터셋:
- 150개 샘플 (각 클래스당 50개)
- 4개 특성: 꽃받침 길이/너비, 꽃잎 길이/너비
- 3개 클래스: Setosa, Versicolor, Virginica
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
from src.mlp import MLP
from src.utils import normalize, one_hot_encode, train_test_split


def load_iris_data():
    """
    간단한 Iris 데이터셋 로드 (sklearn 없이)

    sklearn이 있으면 사용하고, 없으면 더미 데이터 생성
    """
    try:
        from sklearn.datasets import load_iris
        iris = load_iris()
        return iris.data, iris.target, iris.target_names
    except ImportError:
        print("sklearn이 설치되지 않아 더미 데이터를 생성합니다.")
        print("실제 데이터 사용: pip install scikit-learn")

        # 더미 데이터 생성
        np.random.seed(42)
        n_samples_per_class = 50

        # 각 클래스별로 다른 분포의 데이터 생성
        class_0 = np.random.randn(n_samples_per_class, 4) * 0.5 + [5.0, 3.4, 1.5, 0.2]
        class_1 = np.random.randn(n_samples_per_class, 4) * 0.5 + [6.0, 2.8, 4.5, 1.3]
        class_2 = np.random.randn(n_samples_per_class, 4) * 0.5 + [6.5, 3.0, 5.5, 2.0]

        X = np.vstack([class_0, class_1, class_2])
        y = np.hstack([np.zeros(n_samples_per_class),
                       np.ones(n_samples_per_class),
                       np.ones(n_samples_per_class) * 2])

        target_names = ['setosa', 'versicolor', 'virginica']

        return X, y.astype(int), target_names


def main():
    print("=" * 50)
    print("붓꽃 (Iris) 분류 문제")
    print("=" * 50)

    # 데이터 로드
    X, y, target_names = load_iris_data()

    print(f"\n[데이터셋 정보]")
    print(f"샘플 수: {len(X)}")
    print(f"특성 수: {X.shape[1]}")
    print(f"클래스 수: {len(np.unique(y))}")
    print(f"클래스 이름: {target_names}")

    # 데이터 전처리
    X_normalized = normalize(X)
    y_one_hot = one_hot_encode(y, num_classes=3)

    # 학습/테스트 분할
    X_train, X_test, y_train, y_test = train_test_split(
        X_normalized, y_one_hot, test_size=0.2, random_state=42
    )

    print(f"\n학습 데이터: {len(X_train)}개")
    print(f"테스트 데이터: {len(X_test)}개")

    # 신경망 생성
    # 구조: 입력층(4) → 은닉층(8) → 출력층(3)
    print("\n[신경망 구조]")
    print("입력층: 4개 뉴런 (꽃받침 길이, 꽃받침 너비, 꽃잎 길이, 꽃잎 너비)")
    print("은닉층: 8개 뉴런")
    print("출력층: 3개 뉴런 (Setosa, Versicolor, Virginica)")

    model = MLP(layers=[4, 8, 3], activation='sigmoid', learning_rate=0.1)

    # 학습
    print("\n[학습 시작]")
    history = model.train(X_train, y_train, epochs=1000, batch_size=16, verbose=True)

    # 테스트 데이터 평가
    print("\n[평가]")
    test_predictions = model.predict(X_test)
    test_accuracy = model.evaluate(X_test, y_test)

    print(f"테스트 정확도: {test_accuracy * 100:.2f}%")

    # 몇 가지 예측 결과 출력
    print("\n[예측 결과 샘플]")
    print("-" * 60)
    print(f"{'실제 클래스':<15} {'예측 클래스':<15} {'신뢰도':<10} {'결과'}")
    print("-" * 60)

    for i in range(min(10, len(X_test))):
        true_class = np.argmax(y_test[i])
        pred_class = np.argmax(test_predictions[i])
        confidence = test_predictions[i][pred_class]

        true_name = target_names[true_class]
        pred_name = target_names[pred_class]
        result = "✓" if true_class == pred_class else "✗"

        print(f"{true_name:<15} {pred_name:<15} {confidence:.4f}    {result}")

    # 클래스별 정확도
    print("\n[클래스별 정확도]")
    for i, class_name in enumerate(target_names):
        class_mask = np.argmax(y_test, axis=1) == i
        if np.sum(class_mask) > 0:
            class_acc = np.mean(
                np.argmax(test_predictions[class_mask], axis=1) == i
            )
            print(f"{class_name}: {class_acc * 100:.2f}%")

    # 학습 곡선 시각화
    try:
        import matplotlib.pyplot as plt

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))

        # 손실 곡선
        ax1.plot(history['loss'])
        ax1.set_title('Loss over Epochs')
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('Loss')
        ax1.grid(True)

        # 정확도 곡선
        ax2.plot(history['accuracy'])
        ax2.set_title('Accuracy over Epochs')
        ax2.set_xlabel('Epoch')
        ax2.set_ylabel('Accuracy')
        ax2.grid(True)

        plt.tight_layout()
        plt.savefig('iris_learning_curve.png')
        print("\n학습 곡선이 'iris_learning_curve.png'에 저장되었습니다.")

    except ImportError:
        print("\nmatplotlib이 설치되지 않아 그래프를 생성할 수 없습니다.")

    print("\n" + "=" * 50)
    print("붓꽃 분류 완료!")
    print("=" * 50)


if __name__ == "__main__":
    main()
