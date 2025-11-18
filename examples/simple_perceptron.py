"""
단일 퍼셉트론 예제

OR 게이트와 AND 게이트를 단일 퍼셉트론으로 학습합니다.
선형 분류가 가능한 간단한 문제입니다.
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
from src.perceptron import Perceptron, SmoothPerceptron


def test_or_gate():
    """OR 게이트 학습"""
    print("\n" + "=" * 50)
    print("OR 게이트 학습")
    print("=" * 50)

    # OR 게이트 데이터
    X = np.array([[0, 0],
                  [0, 1],
                  [1, 0],
                  [1, 1]])

    y = np.array([0, 1, 1, 1])

    print("\n[OR 진리표]")
    print("0 OR 0 = 0")
    print("0 OR 1 = 1")
    print("1 OR 0 = 1")
    print("1 OR 1 = 1")

    # 퍼셉트론 생성 및 학습
    perceptron = Perceptron(input_size=2, learning_rate=0.1)
    print("\n[학습 중...]")
    perceptron.train(X, y, epochs=20)

    # 예측
    print("\n[예측 결과]")
    print("-" * 30)
    for i in range(len(X)):
        pred = perceptron.predict(X[i])[0]
        actual = y[i]
        result = "✓" if pred == actual else "✗"
        print(f"{X[i]} → {pred} (실제: {actual}) {result}")

    # 가중치 출력
    weights, bias = perceptron.get_weights()
    print(f"\n학습된 가중치: {weights}")
    print(f"학습된 편향: {bias}")


def test_and_gate():
    """AND 게이트 학습"""
    print("\n" + "=" * 50)
    print("AND 게이트 학습")
    print("=" * 50)

    # AND 게이트 데이터
    X = np.array([[0, 0],
                  [0, 1],
                  [1, 0],
                  [1, 1]])

    y = np.array([0, 0, 0, 1])

    print("\n[AND 진리표]")
    print("0 AND 0 = 0")
    print("0 AND 1 = 0")
    print("1 AND 0 = 0")
    print("1 AND 1 = 1")

    # 퍼셉트론 생성 및 학습
    perceptron = Perceptron(input_size=2, learning_rate=0.1)
    print("\n[학습 중...]")
    perceptron.train(X, y, epochs=20)

    # 예측
    print("\n[예측 결과]")
    print("-" * 30)
    for i in range(len(X)):
        pred = perceptron.predict(X[i])[0]
        actual = y[i]
        result = "✓" if pred == actual else "✗"
        print(f"{X[i]} → {pred} (실제: {actual}) {result}")

    # 가중치 출력
    weights, bias = perceptron.get_weights()
    print(f"\n학습된 가중치: {weights}")
    print(f"학습된 편향: {bias}")


def test_smooth_perceptron():
    """부드러운 퍼셉트론 (Sigmoid) 테스트"""
    print("\n" + "=" * 50)
    print("부드러운 퍼셉트론 (Sigmoid 활성화 함수)")
    print("=" * 50)

    # OR 게이트 데이터
    X = np.array([[0, 0],
                  [0, 1],
                  [1, 0],
                  [1, 1]])

    y = np.array([0, 1, 1, 1])

    print("\n시그모이드 퍼셉트론은 0과 1 사이의 확률을 출력합니다.")

    # 부드러운 퍼셉트론 생성 및 학습
    smooth_perceptron = SmoothPerceptron(input_size=2, learning_rate=0.5)
    print("\n[학습 중...]")
    smooth_perceptron.train(X, y, epochs=500)

    # 예측
    print("\n[예측 결과 (확률)]")
    print("-" * 40)
    for i in range(len(X)):
        pred_prob = smooth_perceptron.predict(X[i])[0]
        pred_class = 1 if pred_prob > 0.5 else 0
        actual = y[i]
        result = "✓" if pred_class == actual else "✗"
        print(f"{X[i]} → {pred_prob:.4f} ({'1' if pred_class else '0'}) (실제: {actual}) {result}")


def test_linear_classification():
    """선형 분류 가능한 2D 데이터셋"""
    print("\n" + "=" * 50)
    print("2D 선형 분류 문제")
    print("=" * 50)

    np.random.seed(42)

    # 두 개의 클래스 생성
    # 클래스 0: 원점 근처
    # 클래스 1: (3, 3) 근처
    class_0 = np.random.randn(20, 2) * 0.5 + [0, 0]
    class_1 = np.random.randn(20, 2) * 0.5 + [3, 3]

    X = np.vstack([class_0, class_1])
    y = np.hstack([np.zeros(20), np.ones(20)])

    # 데이터 섞기
    indices = np.random.permutation(len(X))
    X = X[indices]
    y = y[indices]

    print(f"\n총 샘플 수: {len(X)}")
    print(f"클래스 0: 20개, 클래스 1: 20개")

    # 퍼셉트론 학습
    perceptron = SmoothPerceptron(input_size=2, learning_rate=0.1)
    print("\n[학습 중...]")
    perceptron.train(X, y, epochs=100)

    # 정확도 계산
    predictions = (perceptron.predict(X) > 0.5).astype(int)
    accuracy = np.mean(predictions == y)
    print(f"\n정확도: {accuracy * 100:.2f}%")

    # 결정 경계 시각화
    try:
        import matplotlib.pyplot as plt

        # 데이터 포인트 플롯
        plt.figure(figsize=(8, 6))
        plt.scatter(X[y == 0][:, 0], X[y == 0][:, 1],
                   c='blue', label='Class 0', alpha=0.6)
        plt.scatter(X[y == 1][:, 0], X[y == 1][:, 1],
                   c='red', label='Class 1', alpha=0.6)

        # 결정 경계 그리기
        x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
        y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1

        xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.1),
                            np.arange(y_min, y_max, 0.1))

        Z = perceptron.predict(np.c_[xx.ravel(), yy.ravel()])
        Z = Z.reshape(xx.shape)

        plt.contour(xx, yy, Z, levels=[0.5], colors='green',
                   linewidths=2, label='Decision Boundary')

        plt.xlabel('Feature 1')
        plt.ylabel('Feature 2')
        plt.title('Perceptron Decision Boundary')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.savefig('perceptron_decision_boundary.png')
        print("\n결정 경계가 'perceptron_decision_boundary.png'에 저장되었습니다.")

    except ImportError:
        print("\nmatplotlib이 설치되지 않아 그래프를 생성할 수 없습니다.")


def main():
    print("=" * 50)
    print("단일 퍼셉트론 예제")
    print("=" * 50)

    # 각 테스트 실행
    test_or_gate()
    test_and_gate()
    test_smooth_perceptron()
    test_linear_classification()

    print("\n" + "=" * 50)
    print("모든 테스트 완료!")
    print("=" * 50)

    print("\n[참고]")
    print("XOR 게이트는 선형 분류가 불가능하므로")
    print("단일 퍼셉트론으로는 학습할 수 없습니다.")
    print("XOR 문제는 examples/xor_problem.py를 참고하세요.")


if __name__ == "__main__":
    main()
