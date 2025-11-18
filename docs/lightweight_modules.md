# 경량 AI 모델을 만드는 다양한 파이썬 모듈

이 문서는 경량 AI 모델을 개발하기 위한 다양한 파이썬 라이브러리와 프레임워크를 소개합니다.

## 📚 목차

1. [기초 수치 계산 라이브러리](#1-기초-수치-계산-라이브러리)
2. [경량 머신러닝 프레임워크](#2-경량-머신러닝-프레임워크)
3. [모델 최적화 및 변환 도구](#3-모델-최적화-및-변환-도구)
4. [TinyML 및 임베디드 시스템](#4-tinyml-및-임베디드-시스템)
5. [특화된 경량 라이브러리](#5-특화된-경량-라이브러리)
6. [비교 및 선택 가이드](#비교-및-선택-가이드)

---

## 1. 기초 수치 계산 라이브러리

### 1.1 NumPy ⭐ (현재 프로젝트 사용 중)

**특징:**
- 최소한의 의존성으로 신경망 구현 가능
- 행렬 연산의 기본
- 메모리 효율적

**설치:**
```bash
pip install numpy
```

**사용 예시:**
```python
import numpy as np

# 간단한 신경망 레이어
def dense_layer(x, weights, bias):
    return np.dot(x, weights) + bias

# 활성화 함수
def relu(x):
    return np.maximum(0, x)
```

**장점:**
- ✅ 매우 가벼움 (10-20 MB)
- ✅ 순수 Python으로 신경망 이해 가능
- ✅ 교육 목적에 최적

**단점:**
- ❌ GPU 가속 없음
- ❌ 자동 미분 없음
- ❌ 대규모 모델에는 비효율적

---

### 1.2 CuPy

**특징:**
- NumPy 호환 GPU 가속 라이브러리
- CUDA 지원
- NumPy 코드를 거의 수정 없이 GPU에서 실행

**설치:**
```bash
pip install cupy-cuda11x  # CUDA 버전에 맞게 선택
```

**사용 예시:**
```python
import cupy as cp

# NumPy와 동일한 API
x = cp.array([1, 2, 3])
y = cp.array([4, 5, 6])
z = cp.dot(x, y)  # GPU에서 실행
```

**적합한 경우:**
- GPU 가속이 필요한 경우
- NumPy 코드를 최소 변경으로 가속화

---

### 1.3 JAX

**특징:**
- NumPy + 자동 미분 + JIT 컴파일 + GPU/TPU 지원
- 함수형 프로그래밍 스타일
- 고성능 수치 계산

**설치:**
```bash
pip install jax jaxlib
```

**사용 예시:**
```python
import jax.numpy as jnp
from jax import grad, jit

# 자동 미분 가능한 함수
def loss(params, x, y):
    prediction = jnp.dot(x, params)
    return jnp.mean((prediction - y) ** 2)

# 그래디언트 계산
grad_fn = grad(loss)
gradients = grad_fn(params, x, y)

# JIT 컴파일로 가속화
fast_loss = jit(loss)
```

**장점:**
- ✅ NumPy 스타일 + 자동 미분
- ✅ 매우 빠른 성능
- ✅ GPU/TPU 지원

**단점:**
- ❌ 함수형 스타일 학습 필요
- ❌ 상대적으로 높은 학습 곡선

---

## 2. 경량 머신러닝 프레임워크

### 2.1 scikit-learn

**특징:**
- 전통적인 머신러닝 알고리즘 제공
- 간단하고 일관된 API
- CPU 최적화

**설치:**
```bash
pip install scikit-learn
```

**사용 예시:**
```python
from sklearn.neural_network import MLPClassifier
from sklearn.datasets import load_iris

# 데이터 로드
X, y = load_iris(return_X_y=True)

# 신경망 모델
model = MLPClassifier(
    hidden_layer_sizes=(8, 4),
    activation='relu',
    max_iter=1000
)

# 학습
model.fit(X, y)

# 예측
predictions = model.predict(X)
```

**장점:**
- ✅ 사용이 매우 쉬움
- ✅ 전통적인 ML 알고리즘 풍부
- ✅ 작은 데이터셋에 효율적
- ✅ 모델 크기가 작음

**단점:**
- ❌ 딥러닝에는 제한적
- ❌ GPU 지원 없음
- ❌ 대규모 신경망에는 부적합

**적합한 경우:**
- 작은 데이터셋
- 전통적인 ML 알고리즘 필요
- CPU 환경

---

### 2.2 TensorFlow Lite

**특징:**
- 모바일 및 임베디드 기기용 최적화
- 모델 양자화 지원
- 크로스 플랫폼 (Android, iOS, Linux, etc.)

**설치:**
```bash
pip install tensorflow  # TFLite는 TensorFlow에 포함됨
```

**사용 예시:**
```python
import tensorflow as tf

# 기존 모델을 TFLite로 변환
converter = tf.lite.TFLiteConverter.from_keras_model(model)

# 양자화 적용 (모델 크기 감소)
converter.optimizations = [tf.lite.Optimize.DEFAULT]

# 변환
tflite_model = converter.convert()

# 저장
with open('model.tflite', 'wb') as f:
    f.write(tflite_model)

# 추론
interpreter = tf.lite.Interpreter(model_path='model.tflite')
interpreter.allocate_tensors()

# 입력/출력 텐서 정보
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

# 추론 실행
interpreter.set_tensor(input_details[0]['index'], input_data)
interpreter.invoke()
output = interpreter.get_tensor(output_details[0]['index'])
```

**양자화 옵션:**
- **동적 범위 양자화**: 가중치를 int8로 변환
- **전체 정수 양자화**: 모든 연산을 int8로
- **Float16 양자화**: float32를 float16으로

**장점:**
- ✅ 매우 작은 모델 크기 (양자화 시)
- ✅ 모바일/임베디드 최적화
- ✅ 다양한 플랫폼 지원
- ✅ 하드웨어 가속 지원

**단점:**
- ❌ TensorFlow 전체 설치 필요 (변환 시)
- ❌ 일부 연산자 제한

---

### 2.3 PyTorch Mobile

**특징:**
- PyTorch 모델을 모바일에서 실행
- TorchScript로 모델 최적화
- iOS/Android 지원

**설치:**
```bash
pip install torch torchvision
```

**사용 예시:**
```python
import torch
import torch.nn as nn

# 모델 정의
class TinyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(10, 20)
        self.fc2 = nn.Linear(20, 5)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        return self.fc2(x)

model = TinyModel()

# TorchScript로 변환
scripted_model = torch.jit.script(model)

# 모바일 최적화
optimized_model = torch.utils.mobile_optimizer.optimize_for_mobile(scripted_model)

# 저장
optimized_model._save_for_lite_interpreter("model_mobile.ptl")
```

**장점:**
- ✅ PyTorch 생태계 활용
- ✅ 동적 그래프 지원
- ✅ 모바일 최적화

**단점:**
- ❌ 상대적으로 큰 런타임
- ❌ TFLite보다 플랫폼 지원 적음

---

### 2.4 ONNX Runtime

**특징:**
- 다양한 프레임워크 지원 (PyTorch, TensorFlow, scikit-learn)
- 크로스 플랫폼 최적화된 추론 엔진
- CPU, GPU, 임베디드 장치 지원

**설치:**
```bash
pip install onnx onnxruntime
```

**사용 예시:**
```python
import onnx
import onnxruntime as ort
import torch

# PyTorch 모델을 ONNX로 변환
model = TinyModel()
dummy_input = torch.randn(1, 10)

torch.onnx.export(
    model,
    dummy_input,
    "model.onnx",
    input_names=['input'],
    output_names=['output'],
    dynamic_axes={'input': {0: 'batch_size'}}
)

# ONNX Runtime으로 추론
session = ort.InferenceSession("model.onnx")
input_name = session.get_inputs()[0].name
result = session.run(None, {input_name: input_data.numpy()})
```

**양자화:**
```python
from onnxruntime.quantization import quantize_dynamic

# 동적 양자화
quantize_dynamic(
    "model.onnx",
    "model_quantized.onnx",
    weight_type='int8'
)
```

**장점:**
- ✅ 프레임워크 독립적
- ✅ 우수한 추론 성능
- ✅ 다양한 백엔드 지원
- ✅ 모델 표준화

**단점:**
- ❌ 학습 불가 (추론만 가능)
- ❌ 변환 과정 필요

---

## 3. 모델 최적화 및 변환 도구

### 3.1 Neural Network Intelligence (NNI)

**특징:**
- Microsoft의 AutoML 도구킷
- 모델 압축 (pruning, quantization)
- 하이퍼파라미터 최적화

**설치:**
```bash
pip install nni
```

**프루닝 예시:**
```python
from nni.compression.pytorch.pruning import L1NormPruner

# 프루닝 설정
config_list = [{
    'sparsity': 0.5,  # 50% 가중치 제거
    'op_types': ['Conv2d', 'Linear']
}]

pruner = L1NormPruner(model, config_list)
pruned_model = pruner.compress()
```

**양자화 예시:**
```python
from nni.compression.pytorch.quantization import QAT_Quantizer

# 양자화 인식 학습
quantizer = QAT_Quantizer(model, config_list)
quantized_model = quantizer.compress()
```

---

### 3.2 Intel Neural Compressor

**특징:**
- Intel CPU 최적화
- 자동 양자화 튜닝
- 정확도 보존

**설치:**
```bash
pip install neural-compressor
```

**사용 예시:**
```python
from neural_compressor.quantization import fit
from neural_compressor.config import PostTrainingQuantConfig

# 양자화 설정
config = PostTrainingQuantConfig(
    approach='static',
    backend='pytorch'
)

# 양자화 실행
quantized_model = fit(
    model=model,
    conf=config,
    calib_dataloader=calib_dataloader
)
```

---

### 3.3 Distiller (Knowledge Distillation)

**특징:**
- 큰 모델의 지식을 작은 모델로 전이
- 성능 유지하며 모델 크기 감소

**개념:**
```python
import torch
import torch.nn as nn
import torch.nn.functional as F

class DistillationLoss(nn.Module):
    def __init__(self, temperature=3.0, alpha=0.5):
        super().__init__()
        self.temperature = temperature
        self.alpha = alpha

    def forward(self, student_logits, teacher_logits, labels):
        # Soft targets (teacher)
        soft_loss = F.kl_div(
            F.log_softmax(student_logits / self.temperature, dim=1),
            F.softmax(teacher_logits / self.temperature, dim=1),
            reduction='batchmean'
        ) * (self.temperature ** 2)

        # Hard targets (labels)
        hard_loss = F.cross_entropy(student_logits, labels)

        # 결합
        return self.alpha * soft_loss + (1 - self.alpha) * hard_loss

# 사용
teacher_model = LargeModel()  # 큰 모델
student_model = SmallModel()  # 작은 모델
criterion = DistillationLoss()

# 학습
student_logits = student_model(x)
with torch.no_grad():
    teacher_logits = teacher_model(x)

loss = criterion(student_logits, teacher_logits, labels)
```

---

## 4. TinyML 및 임베디드 시스템

### 4.1 Edge Impulse Python SDK

**특징:**
- 임베디드 ML 플랫폼
- 마이크로컨트롤러 배포
- 전체 ML 파이프라인

**설치:**
```bash
pip install edge-impulse-linux
```

**사용 예시:**
```python
from edge_impulse_linux.runner import ImpulseRunner

# 모델 로드
model = ImpulseRunner("model.eim")

# 추론
features = extract_features(data)
result = model.classify(features)
```

**지원 하드웨어:**
- Arduino
- Raspberry Pi
- ESP32
- STM32

---

### 4.2 TensorFlow Lite Micro

**특징:**
- 마이크로컨트롤러용 TFLite
- 매우 작은 메모리 풋프린트 (< 20KB)
- C++ 런타임

**메모리 요구사항:**
- Flash: 16KB - 128KB
- RAM: 2KB - 32KB

**지원 플랫폼:**
- Arduino
- ESP32
- ARM Cortex-M

**예시 (Arduino):**
```cpp
#include <TensorFlowLite.h>

// 모델 로드
const tflite::Model* model = tflite::GetModel(model_data);

// 인터프리터 생성
tflite::MicroInterpreter interpreter(
    model, resolver, tensor_arena, kTensorArenaSize
);

// 추론
interpreter.Invoke();
```

---

### 4.3 MicroML

**특징:**
- 마이크로컨트롤러용 ML 알고리즘
- 전통적인 ML (SVM, Random Forest, etc.)
- 매우 낮은 리소스 요구사항

**설치:**
```bash
pip install emlearn
```

**사용 예시:**
```python
from sklearn.tree import DecisionTreeClassifier
import emlearn

# scikit-learn 모델 학습
model = DecisionTreeClassifier()
model.fit(X_train, y_train)

# C 코드로 변환
c_code = emlearn.convert(model, method='inline')

# model.h 파일 생성
with open('model.h', 'w') as f:
    f.write(c_code)
```

---

### 4.4 μTensor (MicroTensor)

**특징:**
- 마이크로컨트롤러용 신경망 프레임워크
- TensorFlow 모델 변환
- C++ 런타임

**설치:**
```bash
pip install utensor_cgen
```

---

## 5. 특화된 경량 라이브러리

### 5.1 Hummingbird

**특징:**
- 전통적인 ML 모델을 PyTorch/TensorFlow로 변환
- 하드웨어 가속 활용
- scikit-learn, XGBoost, LightGBM 지원

**설치:**
```bash
pip install hummingbird-ml
```

**사용 예시:**
```python
from hummingbird.ml import convert
from sklearn.ensemble import RandomForestClassifier

# scikit-learn 모델
sklearn_model = RandomForestClassifier()
sklearn_model.fit(X, y)

# PyTorch로 변환
pytorch_model = convert(sklearn_model, 'pytorch')

# GPU 추론 가능
predictions = pytorch_model.predict(X)
```

---

### 5.2 sklearn-onnx

**특징:**
- scikit-learn 모델을 ONNX로 변환
- 배포 최적화

**설치:**
```bash
pip install skl2onnx
```

**사용 예시:**
```python
from skl2onnx import convert_sklearn
from skl2onnx.common.data_types import FloatTensorType

# 모델 학습
model = RandomForestClassifier()
model.fit(X, y)

# ONNX로 변환
initial_type = [('float_input', FloatTensorType([None, X.shape[1]]))]
onnx_model = convert_sklearn(model, initial_types=initial_type)

# 저장
with open("model.onnx", "wb") as f:
    f.write(onnx_model.SerializeToString())
```

---

### 5.3 CoreML Tools (Apple)

**특징:**
- iOS/macOS 앱용 ML 모델
- 신경망 가속기 활용
- 다양한 프레임워크 지원

**설치:**
```bash
pip install coremltools
```

**사용 예시:**
```python
import coremltools as ct

# PyTorch 모델 변환
model = torch.jit.trace(pytorch_model, example_input)
coreml_model = ct.convert(
    model,
    inputs=[ct.TensorType(shape=input_shape)]
)

# 저장
coreml_model.save("model.mlmodel")
```

---

### 5.4 Chainer

**특징:**
- Define-by-Run 방식
- 경량 신경망 프레임워크
- 직관적인 API

**설치:**
```bash
pip install chainer
```

**사용 예시:**
```python
import chainer
import chainer.functions as F
import chainer.links as L

class MLP(chainer.Chain):
    def __init__(self):
        super().__init__()
        with self.init_scope():
            self.l1 = L.Linear(784, 100)
            self.l2 = L.Linear(100, 10)

    def forward(self, x):
        h = F.relu(self.l1(x))
        return self.l2(h)

model = MLP()
```

---

### 5.5 mlpack

**특징:**
- C++ ML 라이브러리 (Python 바인딩 제공)
- 매우 빠른 성능
- 메모리 효율적

**설치:**
```bash
pip install mlpack
```

**사용 예시:**
```python
from mlpack import perceptron

# 퍼셉트론 학습
output = perceptron(training=X_train, labels=y_train)
model = output['output_model']

# 예측
predictions = perceptron(input_model=model, test=X_test)
```

---

### 5.6 fastai

**특징:**
- 고수준 PyTorch API
- 빠른 프로토타이핑
- 전이 학습 지원

**설치:**
```bash
pip install fastai
```

**사용 예시:**
```python
from fastai.vision.all import *

# 데이터 로더
dls = ImageDataLoaders.from_folder(path)

# 학습
learn = vision_learner(dls, resnet18, metrics=accuracy)
learn.fine_tune(4)

# 경량화: 모델 압축
learn.export('model.pkl')
```

---

## 비교 및 선택 가이드

### 📊 라이브러리 비교표

| 라이브러리 | 크기 | GPU | 모바일 | 임베디드 | 학습 난이도 | 최적 사용 사례 |
|---------|------|-----|--------|----------|------------|--------------|
| **NumPy** | ⭐⭐⭐⭐⭐ | ❌ | ❌ | ❌ | ⭐ | 교육, 프로토타이핑 |
| **scikit-learn** | ⭐⭐⭐⭐ | ❌ | ❌ | ❌ | ⭐ | 전통 ML, 작은 데이터 |
| **TFLite** | ⭐⭐⭐⭐⭐ | ✅ | ✅ | ✅ | ⭐⭐⭐ | 모바일 앱 |
| **PyTorch Mobile** | ⭐⭐⭐ | ✅ | ✅ | ❌ | ⭐⭐⭐ | 모바일 앱 (PyTorch) |
| **ONNX Runtime** | ⭐⭐⭐⭐ | ✅ | ✅ | ✅ | ⭐⭐ | 크로스 플랫폼 추론 |
| **JAX** | ⭐⭐⭐ | ✅ | ❌ | ❌ | ⭐⭐⭐⭐ | 고성능 연구 |
| **TFLite Micro** | ⭐⭐⭐⭐⭐ | ❌ | ❌ | ✅ | ⭐⭐⭐⭐ | 마이크로컨트롤러 |
| **Edge Impulse** | ⭐⭐⭐⭐ | ❌ | ❌ | ✅ | ⭐⭐ | IoT, TinyML |

---

### 🎯 사용 사례별 권장 라이브러리

#### 1. **교육 및 학습 목적**
```
NumPy → scikit-learn → JAX/PyTorch
```
- NumPy: 신경망 기초 이해
- scikit-learn: 전통적인 ML 알고리즘
- JAX/PyTorch: 고급 딥러닝

#### 2. **모바일 앱 개발**
```
TensorFlow Lite 또는 PyTorch Mobile + ONNX Runtime
```
- TFLite: Android/iOS 모두, 더 성숙함
- PyTorch Mobile: PyTorch 생태계 활용
- ONNX: 프레임워크 독립적

#### 3. **임베디드 시스템 (Raspberry Pi, Jetson)**
```
TensorFlow Lite + ONNX Runtime
```
- TFLite: 하드웨어 가속 지원
- ONNX Runtime: 유연성

#### 4. **마이크로컨트롤러 (Arduino, ESP32)**
```
TensorFlow Lite Micro 또는 Edge Impulse
```
- TFLite Micro: 표준적, 강력함
- Edge Impulse: 전체 파이프라인, 사용 쉬움

#### 5. **웹 브라우저**
```
TensorFlow.js 또는 ONNX.js
```
- TensorFlow.js: 더 많은 기능
- ONNX.js: 경량, 빠름

#### 6. **서버 추론 (CPU)**
```
ONNX Runtime + Intel Neural Compressor
```
- ONNX Runtime: 최고의 CPU 성능
- Neural Compressor: Intel CPU 최적화

#### 7. **연구 및 실험**
```
JAX 또는 PyTorch
```
- JAX: 자동 미분, 고성능
- PyTorch: 유연성, 생태계

---

### 💡 모델 경량화 기법

#### 1. **양자화 (Quantization)**
```
FP32 → FP16 → INT8
```

**메모리 감소:**
- FP32 → FP16: 50% 감소
- FP32 → INT8: 75% 감소

**구현:**
```python
# TensorFlow Lite
converter.optimizations = [tf.lite.Optimize.DEFAULT]

# PyTorch
quantized_model = torch.quantization.quantize_dynamic(
    model, {torch.nn.Linear}, dtype=torch.qint8
)

# ONNX Runtime
quantize_dynamic("model.onnx", "model_int8.onnx")
```

#### 2. **프루닝 (Pruning)**
```
불필요한 가중치 제거 → 모델 희소화 → 압축
```

**구현:**
```python
from torch.nn.utils import prune

# 50% 가중치 프루닝
prune.l1_unstructured(module, name='weight', amount=0.5)
```

#### 3. **지식 증류 (Knowledge Distillation)**
```
큰 Teacher 모델 → 작은 Student 모델로 지식 전이
```

#### 4. **구조 최적화**
```
- Depthwise Separable Convolution
- MobileNet, EfficientNet 아키텍처 사용
- 레이어 수 감소
```

---

### 🚀 시작 가이드

#### 초보자 경로:
```
1. NumPy로 신경망 구조 이해 (현재 프로젝트)
2. scikit-learn으로 전통 ML 학습
3. TensorFlow/PyTorch 기초
4. TFLite로 모바일 배포
```

#### 모바일 개발자 경로:
```
1. TensorFlow/PyTorch로 모델 학습
2. TFLite/PyTorch Mobile로 변환
3. 양자화 적용
4. 앱에 통합
```

#### IoT 개발자 경로:
```
1. 작은 모델 설계 (MobileNet, etc.)
2. Edge Impulse 또는 TFLite Micro 사용
3. 타겟 하드웨어에 배포
4. 최적화 및 튜닝
```

---

### 📦 현재 프로젝트에 적용 가능한 확장

**Tiny AI 프로젝트 확장 아이디어:**

1. **scikit-learn 통합**
   - 전통 ML 알고리즘 비교
   - NumPy 구현과 성능 비교

2. **ONNX 변환**
   - NumPy 모델을 ONNX로 변환
   - 크로스 플랫폼 배포

3. **양자화 데모**
   - 모델 크기 감소 시연
   - 정확도 vs 크기 트레이드오프

4. **TinyML 예제**
   - Arduino/ESP32 배포
   - 실시간 추론 데모

5. **벤치마크**
   - 다양한 라이브러리 성능 비교
   - 메모리, 속도, 정확도 측정

---

## 📚 추가 리소스

### 온라인 코스
- **TinyML Specialization** (Coursera) - Harvard
- **Efficient Deep Learning** (MIT)

### 도서
- "TinyML" by Pete Warden
- "Efficient Processing of Deep Neural Networks" by Vivienne Sze

### 커뮤니티
- TinyML Foundation
- ONNX Community
- Edge Impulse Forum

### 도구
- **Netron**: 모델 시각화 도구
- **Model Analyzer**: 모델 분석 도구
- **Benchmark Tools**: 성능 측정

---

## 🎓 결론

경량 AI 모델 개발을 위한 라이브러리 선택은 다음 요소를 고려해야 합니다:

1. **타겟 플랫폼**: 모바일? 임베디드? 서버?
2. **리소스 제약**: 메모리, CPU/GPU, 배터리
3. **성능 요구사항**: 정확도 vs 속도
4. **개발 경험**: 학습 곡선
5. **생태계**: 커뮤니티, 문서, 지원

**현재 Tiny AI 프로젝트는 NumPy를 사용하여 기초를 다지고 있으며,**
이를 기반으로 필요에 따라 위의 라이브러리들을 점진적으로 도입할 수 있습니다.
