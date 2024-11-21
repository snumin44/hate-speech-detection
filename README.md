# Korean Character-level Hate Speech Detection

- 한국어 **자모 단위**로 욕설 및 혐오 표현을 탐지(Hate Speech Detection)하는 코드입니다.         
- **CNN + GRU** 구조에 기반해 입력 텍스트의 **'hate/clean'** 여부를 분류하는 이진 분류 모델입니다.              

&nbsp;&nbsp;&nbsp;

## 1. Model
- Transformer 기반의 PLM은 실시간의 채팅/댓글을 처리하기에 충분히 빠르지 않을 수 있습니다. 
- 실시간의 데이터를 빠르게 처리하기 위해 다음과 같이 **CNN + GRU** 구조로 모델을 설계했습니다. [\[code\]]()

<p align="center">
<img src="hate_speech_detection_model.PNG" alt="example image" width="500" height="200"/>
</p>
  
- 학습 데이터가 충분할 경우, CNN 층을 늘리거나 GRU를 LSTM으로 대체해 성능을 개선할 수 있습니다.

 
## 2. Character-level Tokenization  
- 자모를 이용한 욕설 및 혐오 표현을 탐지하기 위해 **자모 단위**로 토크나이징을 합니다.
- 한글의 유니코드를 이용해 각 음절에 대응하는 자모를 찾은 후, 기존의 음절을 자모로 대체합니다. [\[code\]]()
- 특수 문자를 이용한 욕설을 처리하기 위해 어휘사전(vocab)에 직접 특수 문자를 추가할 수 있습니다. [\[code\]]()
  

```python
class AddVocab:
    CHOSUNG = {
        'ㄱ':['㉠','㈀'],
        'ㄲ':['刀'],
        'ㄴ':['㉡','㈁'],
        'ㄷ':['㉢','㈂'],
        'ㄸ':[],
        'ㄹ':['㉣','㈃', '己'],
        'ㅁ':['㉤','㈄','口'],
        'ㅂ':['㉥','㈅','廿'],
        'ㅅ':['㉦','㈆', '人'],
        'ㅆ':[],
        'ㅇ':['㉧','㈇'],
        'ㅈ':['㉨','㈈'],
        'ㅉ':[],
        'ㅊ':['㉩','㈉'],
        'ㅋ':['㉪','㈊'],
        'ㅌ':['㉫','㈋'],
        'ㅍ':['㉬','㈌'],
        'ㅎ':['㉭','㈍'],
    }
```


## 3. Training Data 

## 4. Performance
