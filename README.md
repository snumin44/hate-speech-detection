# Korean Character-level Hate Speech Detection

- 한국어 **자모 단위**로 욕설 및 혐오 표현을 탐지(Hate Speech Detection)하는 코드입니다.         
- **CNN + GRU** 구조에 기반해 입력 텍스트의 **'hate/clean'** 여부를 분류하는 이진 분류 모델입니다.              

&nbsp;&nbsp;&nbsp;

## 1. Model
- Transformer 기반의 PLM은 실시간의 채팅/댓글을 처리하기에 충분히 빠르지 않을 수 있습니다. 
- 실시간의 데이터를 빠르게 처리하기 위해 다음과 같이 **CNN + GRU** 구조로 모델을 설계했습니다.

<>

- 학습 데이터가 충분할 경우, CNN의 층을 늘리거나 GRU를 LSTM으로 대체해 성능을 개선할 수 있습니다.   

## 2. Character-level  

## 3. Data 

## 4. Performance
