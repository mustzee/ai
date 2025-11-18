"""
챗봇 기능 테스트
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.chatbot import TinyAIAssistant


def main():
    print("=" * 60)
    print("챗봇 테스트 시작")
    print("=" * 60)
    print()

    # 의도 파일 경로
    intents_file = os.path.join(
        os.path.dirname(os.path.dirname(__file__)),
        'data',
        'intents.json'
    )

    # 챗봇 초기화
    print("1. 챗봇 초기화...")
    assistant = TinyAIAssistant(intents_file=intents_file)
    print("✅ 완료")
    print()

    # 학습
    print("2. 모델 학습...")
    assistant.train(epochs=500)
    print()

    # 테스트 메시지들
    test_messages = [
        "안녕하세요",
        "안녕",
        "반가워",
        "도와줘",
        "고마워",
        "기분이 어때",
        "농담해줘",
        "공부 도와줘",
        "힘들어",
        "잘가"
    ]

    print("3. 대화 테스트")
    print("=" * 60)
    print()

    for msg in test_messages:
        print(f"👤 사용자: {msg}")
        response = assistant.chat(msg)

        # 대화 기록에서 의도와 신뢰도 확인
        last_interaction = assistant.get_conversation_history()[-1]
        intent = last_interaction.get('intent')
        confidence = last_interaction.get('confidence', 0)

        print(f"   [의도: {intent}, 신뢰도: {confidence:.2f}]")
        print(f"🤖 AI: {response}")
        print()

    print("=" * 60)
    print("테스트 완료!")
    print("=" * 60)


if __name__ == "__main__":
    main()
