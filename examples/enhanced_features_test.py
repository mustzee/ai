"""
Tiny AI 개선 기능 테스트

Phase 1 개선사항:
- 컨텍스트 인식 대화
- 신뢰도 기반 불확실성 표현
- 기본 안전 필터
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.chatbot import TinyAIAssistant


def test_context_awareness():
    """컨텍스트 인식 테스트"""
    print("=" * 60)
    print("1. 컨텍스트 인식 대화 테스트")
    print("=" * 60)
    print()

    # 의도 파일 경로
    intents_file = os.path.join(
        os.path.dirname(os.path.dirname(__file__)),
        'data',
        'intents.json'
    )

    # 챗봇 초기화 및 학습
    assistant = TinyAIAssistant(intents_file=intents_file)
    print("모델 학습 중...")
    assistant.train(epochs=500)
    print()

    # 연속 대화 테스트
    conversation = [
        "안녕하세요",
        "공부 도와줘",
        "그거 좀 더 자세히 알려줘",  # 후속 질문
    ]

    print("🔵 연속 대화 테스트:")
    print()
    for msg in conversation:
        print(f"👤 사용자: {msg}")
        response = assistant.chat(msg)
        print(f"🤖 AI: {response}")
        print()

    print("-" * 60)
    print()


def test_confidence_expression():
    """신뢰도 기반 불확실성 표현 테스트"""
    print("=" * 60)
    print("2. 신뢰도 기반 불확실성 표현 테스트")
    print("=" * 60)
    print()

    # 의도 파일 경로
    intents_file = os.path.join(
        os.path.dirname(os.path.dirname(__file__)),
        'data',
        'intents.json'
    )

    # 챗봇 초기화 및 학습
    assistant = TinyAIAssistant(intents_file=intents_file)
    print("모델 학습 중...")
    assistant.train(epochs=500)
    print()

    # 다양한 신뢰도 입력 테스트
    test_cases = [
        ("안녕하세요", "높은 신뢰도 예상"),
        ("뭐 먹을까", "낮은 신뢰도 예상 - 불확실성 표현"),
        ("asdfqwer", "매우 낮은 신뢰도 - 개선된 폴백"),
    ]

    print("🔵 다양한 신뢰도 입력:")
    print()
    for msg, expected in test_cases:
        print(f"👤 사용자: {msg}")
        print(f"   [예상: {expected}]")
        response = assistant.chat(msg)

        # 신뢰도 확인
        last_turn = assistant.get_conversation_history()[-1]
        confidence = last_turn.get('confidence', 0)
        intent = last_turn.get('intent')

        print(f"   [실제 의도: {intent}, 신뢰도: {confidence:.2f}]")
        print(f"🤖 AI: {response}")
        print()

    print("-" * 60)
    print()


def test_safety_filter():
    """안전 필터 테스트"""
    print("=" * 60)
    print("3. 기본 안전 필터 테스트")
    print("=" * 60)
    print()

    # 의도 파일 경로
    intents_file = os.path.join(
        os.path.dirname(os.path.dirname(__file__)),
        'data',
        'intents.json'
    )

    # 챗봇 초기화 및 학습
    assistant = TinyAIAssistant(intents_file=intents_file)
    print("모델 학습 중...")
    assistant.train(epochs=500)
    print()

    # 안전 필터 테스트
    test_cases = [
        ("도와줘", "정상 입력 - 통과"),
        ("살기싫어", "위기 상황 - 필터 발동"),
    ]

    print("🔵 안전 필터 테스트:")
    print()
    for msg, expected in test_cases:
        print(f"👤 사용자: {msg}")
        print(f"   [예상: {expected}]")
        response = assistant.chat(msg)
        print(f"🤖 AI: {response}")
        print()

    print("-" * 60)
    print()


def test_follow_up_detection():
    """후속 질문 감지 테스트"""
    print("=" * 60)
    print("4. 후속 질문 감지 테스트")
    print("=" * 60)
    print()

    # 의도 파일 경로
    intents_file = os.path.join(
        os.path.dirname(os.path.dirname(__file__)),
        'data',
        'intents.json'
    )

    # 챗봇 초기화 및 학습
    assistant = TinyAIAssistant(intents_file=intents_file)
    print("모델 학습 중...")
    assistant.train(epochs=500)
    print()

    # 후속 질문 패턴 테스트
    conversations = [
        ["공부 도와줘", "그럼 어떻게 해?"],
        ["계획 세우고 싶어", "그거 더 알려줘"],
        ["농담해줘", "또 다른 거 있어?"],
    ]

    for conv in conversations:
        print(f"🔵 대화 시퀀스:")
        print()
        for msg in conv:
            print(f"👤 사용자: {msg}")
            response = assistant.chat(msg)

            # 컨텍스트 확인
            state = assistant.conversation_state
            is_follow = assistant._is_follow_up(msg)

            print(f"   [후속 질문: {is_follow}, 이전 의도: {state['last_intent']}]")
            print(f"🤖 AI: {response}")
            print()

        # 대화 기록 초기화
        assistant.clear_history()
        print("-" * 60)
        print()


def test_conversation_state():
    """대화 상태 추적 테스트"""
    print("=" * 60)
    print("5. 대화 상태 추적 테스트")
    print("=" * 60)
    print()

    # 의도 파일 경로
    intents_file = os.path.join(
        os.path.dirname(os.path.dirname(__file__)),
        'data',
        'intents.json'
    )

    # 챗봇 초기화 및 학습
    assistant = TinyAIAssistant(intents_file=intents_file)
    print("모델 학습 중...")
    assistant.train(epochs=500)
    print()

    # 긴 대화 테스트
    conversation = [
        "안녕",
        "도와줘",
        "공부 방법 알려줘",
        "고마워",
        "잘가",
    ]

    print("🔵 대화 상태 추적:")
    print()
    for msg in conversation:
        print(f"👤 사용자: {msg}")
        response = assistant.chat(msg)

        # 상태 출력
        state = assistant.conversation_state
        print(f"   [턴: {state['turn_count']}, 현재 주제: {state['current_topic']}]")
        print(f"🤖 AI: {response}")
        print()

    # 대화 기록 통계
    history = assistant.get_conversation_history()
    print("-" * 60)
    print(f"📊 대화 통계:")
    print(f"  • 총 턴 수: {len(history) // 2}")
    print(f"  • 총 메시지 수: {len(history)}")
    print(f"  • 최종 주제: {assistant.conversation_state['current_topic']}")
    print()


def main():
    """전체 테스트 실행"""
    print("\n")
    print("=" * 60)
    print("Tiny AI Phase 1 개선사항 테스트")
    print("=" * 60)
    print()

    # 1. 컨텍스트 인식
    test_context_awareness()

    # 2. 신뢰도 기반 불확실성
    test_confidence_expression()

    # 3. 안전 필터
    test_safety_filter()

    # 4. 후속 질문 감지
    test_follow_up_detection()

    # 5. 대화 상태 추적
    test_conversation_state()

    print("=" * 60)
    print("✅ 모든 테스트 완료!")
    print("=" * 60)
    print()


if __name__ == "__main__":
    main()
