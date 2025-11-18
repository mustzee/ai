"""
XOR 문제 해결 예제

XOR은 선형 분류가 불가능한 대표적인 문제입니다.
다층 퍼셉트론을 사용하여 XOR 문제를 해결합니다.

XOR 진리표:
0 XOR 0 = 0
0 XOR 1 = 1
1 XOR 0 = 1
1 XOR 1 = 0
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
from src.mlp import MLP


def main():
    print("=" * 50)
    print("XOR 문제 해결하기")
    print("=" * 50)

    # XOR 데이터셋
    X = np.array([[0, 0],
                  [0, 1],
                  [1, 0],
                  [1, 1]])

    y = np.array([[0],
                  [1],
                  [1],
                  [0]])

    print("\n[데이터셋]")
    print("입력 (X):")
    print(X)
    print("\n타겟 (y):")
    print(y.flatten())

    # 신경망 생성
    # 구조: 입력층(2) → 은닉층(4) → 출력층(1)
    print("\n[신경망 구조]")
    print("입력층: 2개 뉴런")
    print("은닉층: 4개 뉴런")
    print("출력층: 1개 뉴런")
    print("활성화 함수: Sigmoid")

    model = MLP(layers=[2, 4, 1], activation='sigmoid', learning_rate=0.5)

    # 학습
    print("\n[학습 시작]")
    history = model.train(X, y, epochs=5000, verbose=True)

    # 예측
    print("\n[학습 완료 - 예측 결과]")
    predictions = model.predict(X)

    print("\n입력 → 예측값 → 실제값")
    print("-" * 40)
    for i in range(len(X)):
        pred = predictions[i][0]
        actual = y[i][0]
        result = "✓" if (pred > 0.5) == actual else "✗"
        print(f"{X[i]} → {pred:.4f} → {actual} {result}")

    # 정확도 계산
    accuracy = np.mean((predictions > 0.5) == y)
    print(f"\n최종 정확도: {accuracy * 100:.2f}%")

    # 학습 곡선 시각화 (matplotlib이 있는 경우)
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
        plt.savefig('xor_learning_curve.png')
        print("\n학습 곡선이 'xor_learning_curve.png'에 저장되었습니다.")

    except ImportError:
        print("\nmatplotlib이 설치되지 않아 그래프를 생성할 수 없습니다.")
        print("설치: pip install matplotlib")

    print("\n" + "=" * 50)
    print("XOR 문제 해결 완료!")
    print("=" * 50)


if __name__ == "__main__":
    main()
