"""
Tiny AI 챗봇 데모

친절하고 논리적인 톤앤매너로 일상 대화를 나누는 챗봇입니다.
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.chatbot import TinyAIAssistant


def print_banner():
    """환영 메시지 출력"""
    print("=" * 60)
    print("🤖 Tiny AI 어시스턴트에 오신 것을 환영합니다!")
    print("=" * 60)
    print()
    print("안녕하세요! 저는 친절하고 논리적으로 대화하는 AI입니다.")
    print("일상적인 대화부터 간단한 문제 해결까지 도와드릴게요.")
    print()
    print("💡 사용 팁:")
    print("  - 자연스럽게 대화하듯이 입력해 주세요")
    print("  - '도와줘', '고마워', '안녕' 등 다양하게 말해보세요")
    print("  - '종료', 'quit', 'exit'를 입력하면 대화를 끝낼 수 있어요")
    print()
    print("=" * 60)
    print()


def main():
    """챗봇 실행"""
    # 배너 출력
    print_banner()

    # AI 어시스턴트 초기화
    intents_file = os.path.join(
        os.path.dirname(os.path.dirname(__file__)),
        'data',
        'intents.json'
    )

    print("🔧 AI 모델 준비 중...")
    assistant = TinyAIAssistant(intents_file=intents_file)

    # 학습
    print("📚 학습 시작...")
    assistant.train(epochs=500)
    print()
    print("✅ 준비 완료! 이제 대화를 시작할 수 있어요.")
    print()

    # 대화 루프
    conversation_count = 0

    while True:
        try:
            # 사용자 입력
            user_input = input("👤 당신: ").strip()

            # 종료 명령 체크
            if user_input.lower() in ['종료', 'quit', 'exit', 'q', '끝']:
                print()
                print("🤖 AI: 대화를 종료합니다. 좋은 하루 보내세요! 👋")
                break

            # 빈 입력 체크
            if not user_input:
                continue

            # 응답 생성
            response = assistant.chat(user_input)

            # 응답 출력
            print(f"🤖 AI: {response}")
            print()

            conversation_count += 1

            # 대화 히스토리 표시 (옵션)
            if conversation_count % 5 == 0:
                print(f"💬 대화 {conversation_count}회 진행 중...")
                print()

        except KeyboardInterrupt:
            print("\n\n🤖 AI: 대화를 중단합니다. 안녕히 가세요!")
            break
        except Exception as e:
            print(f"\n⚠️  오류 발생: {e}")
            print("다시 시도해 주세요.")
            print()

    # 대화 통계
    print()
    print("=" * 60)
    print(f"📊 대화 통계: 총 {conversation_count}회의 대화")
    print("=" * 60)


def demo_conversation():
    """미리 정의된 대화 데모"""
    print("=" * 60)
    print("🎬 데모 대화 시작")
    print("=" * 60)
    print()

    intents_file = os.path.join(
        os.path.dirname(os.path.dirname(__file__)),
        'data',
        'intents.json'
    )

    assistant = TinyAIAssistant(intents_file=intents_file)
    print("학습 중...")
    assistant.train(epochs=500)
    print()

    # 데모 대화
    demo_messages = [
        "안녕하세요!",
        "기분이 어때?",
        "공부 도와줘",
        "농담해줘",
        "고마워",
        "잘가"
    ]

    for msg in demo_messages:
        print(f"👤 사용자: {msg}")
        response = assistant.chat(msg)
        print(f"🤖 AI: {response}")
        print()
        input("(계속하려면 Enter를 누르세요...)")

    print("=" * 60)
    print("데모 종료")
    print("=" * 60)


if __name__ == "__main__":
    import sys

    if len(sys.argv) > 1 and sys.argv[1] == '--demo':
        # 데모 모드
        demo_conversation()
    else:
        # 일반 대화 모드
        main()
