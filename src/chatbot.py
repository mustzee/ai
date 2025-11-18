"""
대화형 AI 챗봇 모듈

친절하고 논리적인 톤앤매너로 일상적인 문제를 해결하는 챗봇입니다.
"""

import numpy as np
import json
import random
from .mlp import MLP
from .text_processing import Tokenizer, TfidfVectorizer


class IntentClassifier:
    """
    의도 분류기

    사용자 입력의 의도를 분류합니다.
    """

    def __init__(self, intents):
        """
        Args:
            intents: 의도 정의 딕셔너리
        """
        self.intents = intents
        self.intent_labels = list(intents.keys())
        self.tokenizer = Tokenizer(max_words=500)
        self.vectorizer = None
        self.model = None

    def prepare_training_data(self):
        """
        학습 데이터 준비

        Returns:
            X: 입력 벡터
            y: 레이블
        """
        texts = []
        labels = []

        for intent_name, intent_data in self.intents.items():
            patterns = intent_data.get('patterns', [])
            for pattern in patterns:
                texts.append(pattern)
                labels.append(self.intent_labels.index(intent_name))

        return texts, labels

    def train(self, epochs=500, learning_rate=0.1):
        """
        의도 분류 모델 학습

        Args:
            epochs: 학습 에포크
            learning_rate: 학습률
        """
        # 데이터 준비
        texts, labels = self.prepare_training_data()

        # 토크나이저 학습
        self.tokenizer.fit(texts)

        # TF-IDF 벡터화
        self.vectorizer = TfidfVectorizer(self.tokenizer)
        self.vectorizer.fit(texts)

        # 벡터 변환
        X = self.vectorizer.transform(texts)

        # 레이블 원-핫 인코딩
        num_intents = len(self.intent_labels)
        y = np.zeros((len(labels), num_intents))
        for i, label in enumerate(labels):
            y[i, label] = 1

        # 신경망 모델 생성
        input_size = X.shape[1]
        print(f"입력 벡터 크기: {input_size}, 샘플 수: {len(X)}")
        self.model = MLP(
            layers=[input_size, 64, 32, num_intents],
            activation='sigmoid',
            learning_rate=learning_rate
        )

        # 학습
        print(f"의도 분류 모델 학습 중... (의도: {num_intents}개)")
        self.model.train(X, y, epochs=epochs, batch_size=4, verbose=True)
        print("학습 완료!")

    def predict(self, text, threshold=0.25):
        """
        의도 예측

        Args:
            text: 입력 텍스트
            threshold: 신뢰도 임계값 (기본값: 0.25)

        Returns:
            (의도, 신뢰도) 튜플
        """
        if self.model is None:
            return None, 0.0

        # 벡터 변환
        X = self.vectorizer.transform([text])

        # 예측
        predictions = self.model.predict(X)[0]
        intent_idx = np.argmax(predictions)
        confidence = predictions[intent_idx]

        if confidence < threshold:
            return None, confidence

        return self.intent_labels[intent_idx], confidence


class ResponseGenerator:
    """
    응답 생성기

    의도에 맞는 친절하고 논리적인 응답을 생성합니다.
    """

    def __init__(self, intents):
        """
        Args:
            intents: 의도 정의 딕셔너리
        """
        self.intents = intents

    def generate(self, intent, context=None):
        """
        응답 생성

        Args:
            intent: 의도 이름
            context: 컨텍스트 정보

        Returns:
            응답 텍스트
        """
        if intent not in self.intents:
            return self.get_fallback_response()

        responses = self.intents[intent].get('responses', [])

        if not responses:
            return self.get_fallback_response()

        # 랜덤하게 응답 선택
        response = random.choice(responses)

        # 컨텍스트로 응답 커스터마이징
        if context:
            response = self.customize_response(response, context)

        return response

    def customize_response(self, response, context):
        """
        컨텍스트에 따라 응답 커스터마이징

        Args:
            response: 기본 응답
            context: 컨텍스트 딕셔너리

        Returns:
            커스터마이징된 응답
        """
        # 템플릿 변수 치환
        for key, value in context.items():
            placeholder = f"{{{key}}}"
            if placeholder in response:
                response = response.replace(placeholder, str(value))

        return response

    def get_fallback_response(self):
        """
        기본 응답 (의도를 파악하지 못한 경우)

        Returns:
            기본 응답
        """
        fallbacks = [
            "죄송하지만 질문을 정확히 이해하지 못했어요. 다시 한번 말씀해 주시겠어요?",
            "음... 이 부분은 제가 아직 잘 모르는 것 같아요. 다른 방식으로 질문해 주시면 도움을 드릴 수 있을 거예요.",
            "잘 이해가 안 되네요. 좀 더 구체적으로 설명해 주실 수 있나요?",
            "제 능력으로는 이 질문에 답변하기 어려울 것 같아요. 다른 질문이 있으시면 기꺼이 도와드리겠습니다!"
        ]
        return random.choice(fallbacks)


class TinyAIAssistant:
    """
    Tiny AI 어시스턴트

    친절하고 논리적인 톤앤매너로 일상 문제를 해결합니다.

    Claude 철학 적용:
    - 컨텍스트 인식: 이전 대화 내용 기억
    - 불확실성 표현: 신뢰도에 따른 솔직한 응답
    - 안전성: 기본 필터링
    """

    def __init__(self, intents_file=None):
        """
        Args:
            intents_file: 의도 정의 JSON 파일 경로
        """
        self.intents = self.load_intents(intents_file) if intents_file else {}
        self.classifier = IntentClassifier(self.intents) if self.intents else None
        self.response_generator = ResponseGenerator(self.intents) if self.intents else None
        self.conversation_history = []
        self.conversation_state = {
            'current_topic': None,
            'last_intent': None,
            'turn_count': 0
        }

    def load_intents(self, filepath):
        """
        의도 정의 파일 로드

        Args:
            filepath: JSON 파일 경로

        Returns:
            의도 딕셔너리
        """
        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                data = json.load(f)
                return data.get('intents', {})
        except Exception as e:
            print(f"의도 파일 로드 실패: {e}")
            return {}

    def train(self, epochs=1000):
        """
        챗봇 학습

        Args:
            epochs: 학습 에포크 (기본값: 1000)
        """
        if self.classifier:
            self.classifier.train(epochs=epochs, learning_rate=0.3)
        else:
            print("학습할 의도 데이터가 없습니다.")

    def _get_conversation_context(self):
        """
        대화 컨텍스트 추출

        Returns:
            컨텍스트 딕셔너리
        """
        context = {
            'has_history': len(self.conversation_history) > 0,
            'turn_count': self.conversation_state['turn_count'],
            'last_intent': self.conversation_state['last_intent'],
            'current_topic': self.conversation_state['current_topic']
        }

        # 최근 3개 턴 요약
        recent_turns = []
        for msg in self.conversation_history[-6:]:  # 최근 3턴 (user + assistant)
            if msg['role'] == 'user':
                recent_turns.append(msg['content'])

        context['recent_inputs'] = recent_turns
        return context

    def _is_follow_up(self, user_input):
        """
        후속 질문인지 판단

        Args:
            user_input: 사용자 입력

        Returns:
            후속 질문 여부
        """
        # 대명사나 지시어가 있으면 후속 질문일 가능성
        follow_up_indicators = [
            '그거', '그게', '그런', '그래서', '그럼', '그리고',
            '그것', '그 방법', '더', '또', '그 외', '다른',
            '좀 더', '계속', '아까', '방금', '그', '그렇게'
        ]

        for indicator in follow_up_indicators:
            if indicator in user_input:
                return True

        return False

    def _apply_safety_filter(self, user_input):
        """
        기본 안전 필터 적용

        Args:
            user_input: 사용자 입력

        Returns:
            (필터 통과 여부, 경고 메시지)
        """
        # 매우 기본적인 안전 필터 (확장 가능)
        unsafe_patterns = [
            '자살', '죽고싶', '살기싫',  # 위기 상황
        ]

        for pattern in unsafe_patterns:
            if pattern in user_input:
                warning = (
                    "힘든 시간을 보내고 계신 것 같아 마음이 아픕니다. "
                    "전문적인 도움이 필요하실 수 있어요.\n\n"
                    "**도움 받을 수 있는 곳:**\n"
                    "• 자살예방 상담전화: 1393\n"
                    "• 정신건강 위기상담: 1577-0199\n"
                    "• 희망의 전화: 129\n\n"
                    "혼자 감당하지 마시고 전문가의 도움을 받아보세요. "
                    "당신은 소중한 사람입니다."
                )
                return False, warning

        return True, None

    def chat(self, user_input):
        """
        사용자 입력에 응답 (컨텍스트 인식 + 불확실성 표현)

        Args:
            user_input: 사용자 입력

        Returns:
            챗봇 응답
        """
        # 1. 안전 필터 적용
        is_safe, safety_message = self._apply_safety_filter(user_input)
        if not is_safe:
            # 안전 경고 메시지 기록 및 반환
            self.conversation_history.append({
                'role': 'user',
                'content': user_input
            })
            self.conversation_history.append({
                'role': 'assistant',
                'content': safety_message,
                'intent': 'safety_warning',
                'confidence': 1.0
            })
            return safety_message

        # 2. 대화 기록 저장
        self.conversation_history.append({
            'role': 'user',
            'content': user_input
        })

        # 3. 의도 분류
        intent, confidence = self.classifier.predict(user_input)

        # 4. 컨텍스트 수집
        conversation_context = self._get_conversation_context()
        is_follow_up = self._is_follow_up(user_input)

        # 5. 응답 생성
        if intent:
            context = {
                'confidence': f"{confidence:.2%}",
                'user_input': user_input,
                'is_follow_up': is_follow_up,
                'last_intent': conversation_context['last_intent']
            }
            response = self.response_generator.generate(intent, context)

            # 6. 신뢰도 기반 불확실성 표현
            response = self._add_confidence_expression(response, confidence, intent)

        else:
            # 낮은 신뢰도: 솔직하게 불확실성 표현
            response = self._get_uncertain_response(confidence, conversation_context)

        # 7. 대화 상태 업데이트
        self.conversation_state['last_intent'] = intent
        self.conversation_state['current_topic'] = intent
        self.conversation_state['turn_count'] += 1

        # 8. 응답 기록
        self.conversation_history.append({
            'role': 'assistant',
            'content': response,
            'intent': intent,
            'confidence': confidence
        })

        return response

    def _add_confidence_expression(self, response, confidence, intent):
        """
        신뢰도에 따라 불확실성 표현 추가

        Args:
            response: 기본 응답
            confidence: 신뢰도 (0.0~1.0)
            intent: 분류된 의도

        Returns:
            불확실성 표현이 추가된 응답
        """
        # 중간 신뢰도 (0.25~0.5): 약간의 불확실성 표현
        if 0.25 <= confidence < 0.5:
            uncertainty_prefixes = [
                "제 이해가 맞다면, ",
                "아마도 이런 의미이신 것 같은데요. ",
                "정확하지 않을 수 있지만, ",
            ]
            prefix = random.choice(uncertainty_prefixes)
            response = prefix + response

        return response

    def _get_uncertain_response(self, confidence, context):
        """
        불확실할 때의 응답 (개선된 폴백)

        Args:
            confidence: 신뢰도
            context: 대화 컨텍스트

        Returns:
            불확실성을 솔직하게 표현한 응답
        """
        # 이전 대화가 있으면 더 친절한 응답
        if context['has_history']:
            responses = [
                "이 부분은 제가 잘 이해하지 못했어요. 다른 방식으로 말씀해 주시겠어요?",
                "음... 정확히 무엇을 원하시는지 확실하지 않네요. 좀 더 구체적으로 설명해 주실 수 있나요?",
                "죄송해요, 이 질문은 제 능력 범위를 벗어나는 것 같아요. 다른 주제로 도와드릴까요?",
            ]
        else:
            responses = [
                "안녕하세요! 무엇을 도와드릴까요? 인사, 감사, 도움 요청, 농담 등 편하게 말씀해 주세요.",
                "반갑습니다! 질문을 이해하지 못했어요. '도와줘', '고마워', '안녕' 같은 일상 대화로 시작해 볼까요?",
            ]

        response = random.choice(responses)

        # 도움이 될 만한 제안 추가
        response += "\n\n💡 **이런 것들을 도와드릴 수 있어요:**\n"
        response += "• 인사 나누기\n"
        response += "• 일상 대화\n"
        response += "• 공부/계획/문제해결 조언\n"
        response += "• 동기부여와 격려\n"

        return response

    def get_conversation_history(self):
        """
        대화 기록 반환

        Returns:
            대화 기록 리스트
        """
        return self.conversation_history

    def clear_history(self):
        """대화 기록 및 상태 초기화"""
        self.conversation_history = []
        self.conversation_state = {
            'current_topic': None,
            'last_intent': None,
            'turn_count': 0
        }

    def save_model(self, filepath):
        """
        모델 저장

        Args:
            filepath: 저장 경로
        """
        if self.classifier and self.classifier.model:
            self.classifier.model.save_weights(filepath)
            print(f"모델이 {filepath}에 저장되었습니다.")

    def load_model(self, filepath):
        """
        모델 로드

        Args:
            filepath: 로드 경로
        """
        if self.classifier and self.classifier.model:
            self.classifier.model.load_weights(filepath)
            print(f"모델이 {filepath}에서 로드되었습니다.")


def create_sample_intents():
    """
    샘플 의도 데이터 생성

    Returns:
        의도 딕셔너리
    """
    intents = {
        "greeting": {
            "patterns": [
                "안녕",
                "안녕하세요",
                "하이",
                "헬로",
                "반가워",
                "좋은 아침",
                "좋은 저녁",
                "만나서 반가워"
            ],
            "responses": [
                "안녕하세요! 반갑습니다. 무엇을 도와드릴까요?",
                "안녕하세요! 오늘 하루는 어떠신가요? 도움이 필요하시면 말씀해 주세요.",
                "반갑습니다! 어떤 도움이 필요하신가요?",
                "안녕하세요! 편하게 질문해 주세요. 최선을 다해 도와드리겠습니다."
            ]
        },
        "goodbye": {
            "patterns": [
                "안녕",
                "잘가",
                "바이",
                "또 봐",
                "나중에 봐",
                "좋은 하루",
                "수고해"
            ],
            "responses": [
                "안녕히 가세요! 좋은 하루 되세요!",
                "또 만나요! 언제든 도움이 필요하시면 찾아주세요.",
                "즐거운 하루 보내세요! 다음에 또 뵙겠습니다.",
                "좋은 시간 되세요! 필요하시면 언제든 돌아와 주세요."
            ]
        },
        "thanks": {
            "patterns": [
                "고마워",
                "감사합니다",
                "감사해",
                "고맙습니다",
                "땡큐",
                "도움이 됐어"
            ],
            "responses": [
                "천만에요! 도움이 되었다니 기쁩니다.",
                "별말씀을요! 언제든 도와드릴 준비가 되어 있어요.",
                "기꺼이 도와드렸습니다! 다른 질문 있으시면 언제든지요.",
                "감사하다니 제가 더 기쁩니다. 또 궁금한 점 있으시면 말씀해 주세요!"
            ]
        },
        "help_request": {
            "patterns": [
                "도와줘",
                "도움이 필요해",
                "문제가 있어",
                "어떻게 해야 해",
                "모르겠어",
                "궁금해",
                "알려줘"
            ],
            "responses": [
                "물론이죠! 무엇을 도와드릴까요? 구체적으로 말씀해 주시면 더 정확히 도와드릴 수 있어요.",
                "기꺼이 도와드리겠습니다! 어떤 부분이 궁금하신가요?",
                "네, 도와드릴게요! 문제를 자세히 설명해 주시면 함께 해결책을 찾아보겠습니다.",
                "걱정 마세요! 차근차근 함께 해결해 봅시다. 무엇이 문제인가요?"
            ]
        },
        "time_question": {
            "patterns": [
                "지금 몇 시야",
                "시간 알려줘",
                "현재 시간",
                "몇 시니",
                "시간 좀 알려줘"
            ],
            "responses": [
                "죄송하지만 저는 현재 시간 정보에 접근할 수 없어요. 기기의 시계를 확인해 보시는 건 어떨까요?",
                "안타깝게도 실시간 정보는 제공할 수 없어요. 스마트폰이나 컴퓨터의 시계를 확인해 주세요!",
                "현재 시간은 제가 알 수 없지만, 기기의 시계 앱을 확인하시면 정확한 시간을 보실 수 있어요."
            ]
        },
        "how_are_you": {
            "patterns": [
                "어때",
                "잘 지내",
                "기분이 어때",
                "요즘 어때",
                "잘 있어"
            ],
            "responses": [
                "저는 항상 좋아요! 감사합니다. 당신은 어떠신가요?",
                "잘 지내고 있어요! 오늘 당신의 기분은 어떠신가요?",
                "좋습니다! 질문에 답변할 수 있어서 행복해요. 당신은 어떤가요?",
                "훌륭해요! 당신을 도울 수 있어서 기쁩니다. 오늘 하루는 어떠셨나요?"
            ]
        },
        "weather": {
            "patterns": [
                "날씨 어때",
                "오늘 날씨",
                "비 와",
                "날씨 알려줘",
                "맑아"
            ],
            "responses": [
                "죄송하지만 실시간 날씨 정보는 제공할 수 없어요. 날씨 앱이나 웹사이트를 확인해 보시는 건 어떨까요?",
                "날씨 정보는 제가 접근할 수 없는 부분이에요. 기상청 웹사이트나 날씨 앱을 추천드려요!",
                "안타깝게도 실시간 날씨 데이터는 없지만, 스마트폰의 날씨 위젯을 확인하시면 정확한 정보를 보실 수 있어요."
            ]
        },
        "joke": {
            "patterns": [
                "재미있는 얘기해줘",
                "농담해줘",
                "웃긴 얘기",
                "심심해",
                "재밌게 해줘"
            ],
            "responses": [
                "AI와 인간의 차이점이 뭘까요? 인간은 실수를 하고, AI는... 더 빠르게 실수를 하죠! 😄",
                "프로그래머가 컴퓨터를 고치려고 했는데... 결국 껐다 켰어요. 만능 해결책이죠!",
                "0과 1이 싸웠는데요... 2진법으로 끝났대요! (죄송해요, 컴퓨터 농담이에요 😅)",
                "제가 농담을 잘 못해서 죄송해요! 하지만 열심히 배우고 있어요. 다른 도움이 필요하신가요?"
            ]
        },
        "compliment": {
            "patterns": [
                "똑똑하다",
                "잘한다",
                "훌륭해",
                "대단해",
                "최고야"
            ],
            "responses": [
                "감사합니다! 칭찬해 주셔서 힘이 나요. 더 열심히 도와드리겠습니다!",
                "정말 감사합니다! 당신도 아주 훌륭한 분이시네요. 😊",
                "과찬이세요! 하지만 기분은 좋네요. 계속 발전하는 AI가 되겠습니다!",
                "고마워요! 이런 격려가 있어서 더 열심히 학습할 수 있어요."
            ]
        },
        "problem_solving": {
            "patterns": [
                "문제 풀어줘",
                "해결 방법",
                "어떻게 하면 돼",
                "방법 알려줘",
                "해결책",
                "조언해줘"
            ],
            "responses": [
                "문제를 함께 해결해 봅시다! 먼저 상황을 자세히 설명해 주시겠어요? 그러면 가능한 해결책을 생각해볼게요.",
                "좋아요! 문제 해결을 도와드리겠습니다. 구체적으로 어떤 문제인지 말씀해 주세요.",
                "물론이죠! 논리적으로 접근해 봅시다. 1) 문제가 무엇인지, 2) 원하는 결과가 무엇인지 말씀해 주세요.",
                "함께 해결책을 찾아봅시다! 문제의 세부사항을 알려주시면 단계별로 접근해 보겠습니다."
            ]
        }
    }

    return {"intents": intents}
