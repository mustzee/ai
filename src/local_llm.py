"""
Local LLM Module - Offline Language Model Serving
로컬 언어 모델 서빙 (인터넷 연결 없이 사용 가능)

This module provides interface for running large language models locally
using llama.cpp backend. Supports GGUF format models.

Supported Models:
- Llama 2/3 (7B, 13B, 70B)
- Mistral (7B)
- Phi-2/3
- TinyLlama
- Qwen
- Any GGUF format model

Usage:
    from src.local_llm import LocalLLM

    # Initialize with model path
    llm = LocalLLM(model_path="models/llama-2-7b-chat.Q4_K_M.gguf")

    # Generate text
    response = llm.generate("Tell me about AI")
    print(response)
"""

import os
from typing import List, Dict, Optional, Union
import warnings


class LocalLLM:
    """
    로컬 LLM 인터페이스

    llama.cpp를 사용하여 CPU/GPU에서 GGUF 모델을 실행합니다.
    인터넷 연결 없이 완전히 로컬에서 동작합니다.
    """

    def __init__(
        self,
        model_path: str,
        n_ctx: int = 2048,
        n_threads: Optional[int] = None,
        n_gpu_layers: int = 0,
        verbose: bool = False
    ):
        """
        로컬 LLM 초기화

        Parameters:
        -----------
        model_path : str
            GGUF 형식의 모델 파일 경로
            예: "models/llama-2-7b-chat.Q4_K_M.gguf"

        n_ctx : int, default=2048
            컨텍스트 길이 (토큰 수)

        n_threads : int, optional
            사용할 CPU 스레드 수 (None이면 자동)

        n_gpu_layers : int, default=0
            GPU로 오프로드할 레이어 수
            0 = CPU만 사용, 35+ = 대부분 GPU 사용

        verbose : bool, default=False
            상세 로그 출력 여부
        """
        self.model_path = model_path
        self.n_ctx = n_ctx
        self.n_threads = n_threads
        self.n_gpu_layers = n_gpu_layers
        self.verbose = verbose
        self.llm = None

        # Check if model file exists
        if not os.path.exists(model_path):
            raise FileNotFoundError(
                f"모델 파일을 찾을 수 없습니다: {model_path}\n"
                f"모델을 다운로드하려면:\n"
                f"1. https://huggingface.co/TheBloke 방문\n"
                f"2. GGUF 모델 검색 (예: Llama-2-7B-Chat-GGUF)\n"
                f"3. Q4_K_M 또는 Q5_K_M 파일 다운로드\n"
                f"4. {os.path.dirname(model_path)} 폴더에 저장"
            )

        self._load_model()

    def _load_model(self):
        """모델을 메모리에 로드"""
        try:
            from llama_cpp import Llama
        except ImportError:
            raise ImportError(
                "llama-cpp-python이 설치되지 않았습니다.\n"
                "설치: pip install llama-cpp-python\n"
                "또는 GPU 지원: CMAKE_ARGS=\"-DLLAMA_CUBLAS=on\" pip install llama-cpp-python"
            )

        if self.verbose:
            print(f"모델 로딩 중: {self.model_path}")
            print(f"- 컨텍스트 길이: {self.n_ctx}")
            print(f"- GPU 레이어: {self.n_gpu_layers}")

        self.llm = Llama(
            model_path=self.model_path,
            n_ctx=self.n_ctx,
            n_threads=self.n_threads,
            n_gpu_layers=self.n_gpu_layers,
            verbose=self.verbose
        )

        if self.verbose:
            print("✓ 모델 로딩 완료!")

    def generate(
        self,
        prompt: str,
        max_tokens: int = 256,
        temperature: float = 0.7,
        top_p: float = 0.9,
        top_k: int = 40,
        repeat_penalty: float = 1.1,
        stop: Optional[List[str]] = None,
        stream: bool = False
    ) -> Union[str, 'Iterator[str]']:
        """
        텍스트 생성

        Parameters:
        -----------
        prompt : str
            입력 프롬프트

        max_tokens : int, default=256
            생성할 최대 토큰 수

        temperature : float, default=0.7
            샘플링 온도 (0.0 = 결정론적, 1.0 = 창의적)

        top_p : float, default=0.9
            Nucleus sampling (0.0 ~ 1.0)

        top_k : int, default=40
            Top-K sampling

        repeat_penalty : float, default=1.1
            반복 패널티 (1.0 = 없음, >1.0 = 반복 억제)

        stop : List[str], optional
            생성 중단 문자열 리스트

        stream : bool, default=False
            스트리밍 모드 (True면 generator 반환)

        Returns:
        --------
        str or Iterator[str]
            생성된 텍스트 (stream=False) 또는 토큰 제너레이터 (stream=True)
        """
        if self.llm is None:
            raise RuntimeError("모델이 로드되지 않았습니다.")

        output = self.llm(
            prompt,
            max_tokens=max_tokens,
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
            repeat_penalty=repeat_penalty,
            stop=stop or [],
            stream=stream,
            echo=False
        )

        if stream:
            # Return generator for streaming
            def token_generator():
                for chunk in output:
                    if 'choices' in chunk and len(chunk['choices']) > 0:
                        text = chunk['choices'][0].get('text', '')
                        if text:
                            yield text
            return token_generator()
        else:
            # Return complete text
            if 'choices' in output and len(output['choices']) > 0:
                return output['choices'][0]['text']
            return ""

    def chat(
        self,
        messages: List[Dict[str, str]],
        max_tokens: int = 256,
        temperature: float = 0.7,
        **kwargs
    ) -> str:
        """
        채팅 형식으로 대화

        Parameters:
        -----------
        messages : List[Dict[str, str]]
            메시지 리스트
            예: [
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": "Hello!"},
                {"role": "assistant", "content": "Hi! How can I help?"},
                {"role": "user", "content": "Tell me about AI."}
            ]

        max_tokens : int, default=256
            생성할 최대 토큰 수

        temperature : float, default=0.7
            샘플링 온도

        Returns:
        --------
        str
            어시스턴트의 응답
        """
        if self.llm is None:
            raise RuntimeError("모델이 로드되지 않았습니다.")

        try:
            # Try using chat completion if available
            output = self.llm.create_chat_completion(
                messages=messages,
                max_tokens=max_tokens,
                temperature=temperature,
                **kwargs
            )

            if 'choices' in output and len(output['choices']) > 0:
                return output['choices'][0]['message']['content']
            return ""

        except AttributeError:
            # Fallback to manual prompt formatting
            prompt = self._format_chat_prompt(messages)
            return self.generate(
                prompt,
                max_tokens=max_tokens,
                temperature=temperature,
                **kwargs
            )

    def _format_chat_prompt(self, messages: List[Dict[str, str]]) -> str:
        """
        메시지를 프롬프트 문자열로 변환
        (Llama-2 chat format)
        """
        prompt = ""

        for msg in messages:
            role = msg.get("role", "user")
            content = msg.get("content", "")

            if role == "system":
                prompt += f"<<SYS>>\n{content}\n<</SYS>>\n\n"
            elif role == "user":
                prompt += f"[INST] {content} [/INST] "
            elif role == "assistant":
                prompt += f"{content} "

        return prompt.strip()

    def embed(self, text: str) -> List[float]:
        """
        텍스트 임베딩 생성

        Parameters:
        -----------
        text : str
            임베딩할 텍스트

        Returns:
        --------
        List[float]
            임베딩 벡터
        """
        if self.llm is None:
            raise RuntimeError("모델이 로드되지 않았습니다.")

        try:
            embedding = self.llm.embed(text)
            return embedding
        except Exception as e:
            warnings.warn(f"임베딩 생성 실패: {e}")
            return []

    def get_model_info(self) -> Dict[str, any]:
        """
        모델 정보 반환

        Returns:
        --------
        Dict
            모델 메타데이터
        """
        return {
            "model_path": self.model_path,
            "context_length": self.n_ctx,
            "gpu_layers": self.n_gpu_layers,
            "threads": self.n_threads,
            "loaded": self.llm is not None
        }

    def unload(self):
        """모델을 메모리에서 언로드"""
        if self.llm is not None:
            del self.llm
            self.llm = None
            if self.verbose:
                print("✓ 모델 언로드 완료")

    def __del__(self):
        """소멸자: 자동으로 모델 언로드"""
        self.unload()

    def __repr__(self):
        return f"LocalLLM(model='{os.path.basename(self.model_path)}', ctx={self.n_ctx})"


class LocalLLMServer:
    """
    간단한 로컬 LLM 서버

    여러 모델을 관리하고 API 형태로 제공합니다.
    """

    def __init__(self, verbose: bool = False):
        """
        서버 초기화

        Parameters:
        -----------
        verbose : bool, default=False
            상세 로그 출력 여부
        """
        self.models: Dict[str, LocalLLM] = {}
        self.verbose = verbose

    def load_model(
        self,
        name: str,
        model_path: str,
        **kwargs
    ) -> None:
        """
        모델 로드

        Parameters:
        -----------
        name : str
            모델 이름 (식별자)

        model_path : str
            모델 파일 경로

        **kwargs
            LocalLLM에 전달할 추가 인자
        """
        if name in self.models:
            if self.verbose:
                print(f"경고: '{name}' 모델이 이미 로드되어 있습니다. 교체합니다.")
            self.models[name].unload()

        self.models[name] = LocalLLM(
            model_path=model_path,
            verbose=self.verbose,
            **kwargs
        )

        if self.verbose:
            print(f"✓ '{name}' 모델 로드 완료")

    def unload_model(self, name: str) -> None:
        """
        모델 언로드

        Parameters:
        -----------
        name : str
            모델 이름
        """
        if name in self.models:
            self.models[name].unload()
            del self.models[name]
            if self.verbose:
                print(f"✓ '{name}' 모델 언로드 완료")
        else:
            raise KeyError(f"모델 '{name}'을 찾을 수 없습니다.")

    def get_model(self, name: str) -> LocalLLM:
        """
        모델 가져오기

        Parameters:
        -----------
        name : str
            모델 이름

        Returns:
        --------
        LocalLLM
            로드된 모델 인스턴스
        """
        if name not in self.models:
            raise KeyError(
                f"모델 '{name}'을 찾을 수 없습니다.\n"
                f"사용 가능한 모델: {list(self.models.keys())}"
            )
        return self.models[name]

    def list_models(self) -> List[str]:
        """
        로드된 모델 목록 반환

        Returns:
        --------
        List[str]
            모델 이름 리스트
        """
        return list(self.models.keys())

    def generate(self, model_name: str, prompt: str, **kwargs) -> str:
        """
        지정된 모델로 텍스트 생성

        Parameters:
        -----------
        model_name : str
            사용할 모델 이름

        prompt : str
            입력 프롬프트

        **kwargs
            generate()에 전달할 추가 인자

        Returns:
        --------
        str
            생성된 텍스트
        """
        model = self.get_model(model_name)
        return model.generate(prompt, **kwargs)

    def chat(self, model_name: str, messages: List[Dict[str, str]], **kwargs) -> str:
        """
        지정된 모델로 채팅

        Parameters:
        -----------
        model_name : str
            사용할 모델 이름

        messages : List[Dict[str, str]]
            메시지 리스트

        **kwargs
            chat()에 전달할 추가 인자

        Returns:
        --------
        str
            어시스턴트의 응답
        """
        model = self.get_model(model_name)
        return model.chat(messages, **kwargs)

    def __repr__(self):
        return f"LocalLLMServer(models={list(self.models.keys())})"


# Utility functions
def download_model_guide():
    """
    모델 다운로드 가이드 출력
    """
    guide = """
    ===================================
    로컬 LLM 모델 다운로드 가이드
    ===================================

    1. HuggingFace에서 GGUF 모델 다운로드:
       https://huggingface.co/TheBloke

    2. 추천 모델:

       초보자용 (작고 빠름):
       - TinyLlama-1.1B-Chat-v1.0-GGUF (Q4_K_M)
         https://huggingface.co/TheBloke/TinyLlama-1.1B-Chat-v1.0-GGUF

       중급 (균형잡힘):
       - Llama-2-7B-Chat-GGUF (Q4_K_M)
         https://huggingface.co/TheBloke/Llama-2-7B-Chat-GGUF

       - Mistral-7B-Instruct-v0.2-GGUF (Q4_K_M)
         https://huggingface.co/TheBloke/Mistral-7B-Instruct-v0.2-GGUF

       고급 (성능 우선):
       - Llama-2-13B-Chat-GGUF (Q5_K_M)
         https://huggingface.co/TheBloke/Llama-2-13B-Chat-GGUF

    3. 다운로드 방법:
       - 브라우저에서 직접 다운로드, 또는
       - wget 사용:
         wget https://huggingface.co/.../model.gguf

    4. 저장 위치:
       models/ 폴더에 저장 (폴더가 없으면 생성)

       mkdir -p models
       mv downloaded-model.gguf models/

    5. 양자화 수준 (파일 크기 vs 품질):
       - Q4_K_M: 추천 (좋은 균형)
       - Q5_K_M: 더 나은 품질
       - Q6_K: 최고 품질 (크기 큼)
       - Q3_K_M: 작은 크기 (품질 저하)

    ===================================
    """
    print(guide)


if __name__ == "__main__":
    # 다운로드 가이드 출력
    download_model_guide()

    # 사용 예제
    print("\n사용 예제:")
    print("-" * 50)
    print("""
    from src.local_llm import LocalLLM

    # 모델 로드
    llm = LocalLLM(
        model_path="models/tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf",
        n_ctx=2048,
        verbose=True
    )

    # 텍스트 생성
    response = llm.generate(
        "Tell me about artificial intelligence",
        max_tokens=200,
        temperature=0.7
    )
    print(response)

    # 채팅
    messages = [
        {"role": "system", "content": "You are a helpful AI assistant."},
        {"role": "user", "content": "What is machine learning?"}
    ]
    response = llm.chat(messages)
    print(response)
    """)
