# Korean Character-level Hate Speech Detection

- **자모 단위**로 한국어 욕설 및 혐오 표현을 탐지하는 모델을 학습하는 코드입니다.         
- **CNN + GRU** 구조에 기반해 입력 텍스트의 **'Hate/Clean'** 여부를 판별하는 이진 분류 모델입니다.              

&nbsp;&nbsp;&nbsp;

## 1. Model
- Transformer 기반의 PLM은 실시간의 채팅/댓글을 처리하기에 충분히 빠르지 않을 수 있습니다. 
- 실시간의 데이터를 빠르게 처리하기 위해 다음과 같이 **CNN + GRU** 구조로 모델을 설계했습니다. [\[code\]](https://github.com/snumin44/hate-speech-detection/blob/main/src/model.py)

<p align="center">
<img src="hate_speech_detection_model.PNG" alt="example image" width="500" height="200"/>
</p>
   
## 2. Character-level Tokenization  
- 자모를 이용한 욕설 및 혐오 표현을 탐지하기 위해 **자모 단위**로 토크나이징을 합니다.
- 한글의 유니코드를 이용해 각 음절에 대응하는 자모를 찾은 후, 기존의 음절을 자모로 대체합니다. [\[code\]](https://github.com/snumin44/hate-speech-detection/blob/main/utils/utils.py)
- 특수 문자를 이용한 욕설을 처리하기 위해 어휘사전(vocab)에 직접 특수 문자를 추가할 수 있습니다. [\[code\]](https://github.com/snumin44/hate-speech-detection/blob/main/utils/vocab.py)
  

```python
# 형태가 유사한 특수문자를 각 초성과 동일한 숫자로 인코딩 
class AddVocab:
    CHOSUNG = { ...
        'ㄹ':['㉣','㈃', '己'],
        'ㅁ':['㉤','㈄','口'],
        'ㅂ':['㉥','㈅','廿'],
        'ㅅ':['㉦','㈆', '人'], 
    ... }
```

## 3. Dataset 
- 모델 학습과 평가를 위해 다음 한국어 Hate Speech Detection 데이터 셋을 사용했습니다.
  - [BEEP!](https://github.com/kocohub/korean-hate-speech)
  - [욕설 감지 데이터셋](https://github.com/2runo/Curse-detection-data)
  - [Korean UnSmile Dataset](https://github.com/smilegate-ai/korean_unsmile_dataset)
  - 개인적으로 수집한 게임 도메인 댓글 등

- 표현의 자유를 보장하기 위해 **(1) 감탄사, (2) 자신을 향한 욕설**에 해당하는 샘플은 레이블을 직접 수정했습니다.
```
(원본) 이 게임을 돈 주고 산 내가 진짜 ㅄ이지ㅋㅋ → Hate Speech 'True'
(수정) 이 게임을 돈 주고 산 내가 진짜 ㅄ이지ㅋㅋ → Hate Speech 'False'
```
- 학습, 검증, 테스트에 사용한 샘플의 개수는 다음과 같습니다.
  - Train : 36,000 / Validation : 2,000 / Test : 2,000

&nbsp;&nbsp; (※학습에 사용한 데이터 셋은 제공하지 않습니다.)

## 4. Performance
- 학습에 적용한 하이퍼파라미터는 다음과 같습니다.
  - Embedding Dimension : 100
  - Number of Kernels : 100
  - Kernel Sizes : 3, 4, 5
  - Stride : 1
  - Hidden Dimension of GRU : 100
  - Epochs : 30
  - Dropout : 0.2
  - Batch Size : 128
  - Learning Rate : 1e-3

- 실험 결과는 다음과 같습니다.

|Metric|Performance|
|:---:|:---:|
|Precision|81.52|
|Recall|79.49|
|Accuracy|81.56|

- Hate Speech Detection에서는 모델이 Hate 으로 예측한 샘플 중 실제 Hate 의 비율인 **Precision** 이 중요합니다.
- Precision이 높다는 것은 모델이 Clean 을 잘못 분류하지 않고 Hate 만 정밀하게 찾아냈다는 것을 의미합니다. 
- 학습 데이터가 충분할 경우, CNN 층을 늘리거나 GRU를 LSTM으로 대체해 성능을 개선할 수 있습니다.

## 5. Implementation

**(1) 데이터 준비**
- 'text'와 'label' 헤더를 가지는 csv 파일 데이터 셋을 준비합니다.
- 학습, 검증, 테스트를 위한 데이터 셋이 각각 필요합니다. 
```python
text, label
이 게임을 돈 주고 산 내가 진짜 ㅄ이지ㅋㅋ, 0
```
- csv 파일의 데이터는 Pandas 라이브러리를 통해 로드됩니다. [\[code\]](https://github.com/snumin44/hate-speech-detection/blob/main/src/data_loader.py)
```python
@classmethod
def load_csv_data(cls, path, sep='\t', remove_jongsung=True, remove_blank=True):
    #load data
    datasets = pd.read_csv(path, sep=sep)
    texts, label = list(datasets['text']), list(datasets['label'])
```
- 특정 초성으로 처리하고 싶은 특수 문자 있다면 다음 코드에 직접 추가하면 됩니다. [\[code\]](https://github.com/snumin44/hate-speech-detection/blob/main/utils/vocab.py)
```python
# 형태가 유사한 특수문자를 각 초성과 동일한 숫자로 인코딩 
class AddVocab:
    CHOSUNG = { ...
        'ㄹ':['㉣','㈃', '己'],
        'ㅁ':['㉤','㈄','口'],
        'ㅂ':['㉥','㈅','廿'],
        'ㅅ':['㉦','㈆', '人'], 
    ... }
```

**(2) 모델 학습**
- train 디렉토리의 쉘 스크립트를 실행해 모델을 학습할 수 있습니다.
- 쉘 스크립트에서 데이터의 경로 및 하이퍼 파라미터를 직접 변경할 수 있습니다. 
```
cd train
sh run_train.sh
```

**(3) 모델 평가**
- evaluate 디렉토리의 쉘 스크립트를 실행해 학습한 모델을 평가할 수 있습니다.
- 모델 구조에 관한 하이퍼 파라미터가 학습시와 다를 경우 에러가 발생합니다.
```
cd evaluate
sh run_evaluate.sh
``` 

**(4) 추론**
- evaluate 디렉토리의 inference.py를 실행해 학습한 모델을 직접 평가할 수 있습니다.
- asyncio 라이브러리를 이용해 비동기 방식으로 입력 문장을 처리하도록 구현했습니다.
```
cd evaluate
python3 inference.py
``` 
## 6. Demo
- 이상의 모델로 만든 데모는 다음과 같습니다. inference.py 로 동작합니다.
- **(1) 감탄사, (2) 자신을 향한 욕설**에 해당하는 문장은 필터링하지 않는다는 사실을 확인할 수 있습니다.

<p align="center">
<img src="hatespeech.gif" width="480" height="280" alt="Hate Speech Detection (Demo)">
</p>
