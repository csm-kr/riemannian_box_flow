# PROJECT.md

## 목표
MNIST digits 0-9를 사용하는 **box flow toy example** 구현.  
검은색 224x224 canvas 위에 숫자 0~9를 각각 1개씩 배치하고,  
10개의 init box state에서 10개의 target box state로 이동하는 flow matching 학습을 수행한다.

## 범위
- box flow / trajectory를 보기 쉬운 toy setting 구축
- Euclidean / Riemannian box space 비교를 위한 최소 실험 환경
- index별 `b0 -> b1` trajectory 시각화 및 GIF 저장
- flow matching만 다룬다 (classification loss, Hungarian matching 제외)
- query 수 10개 고정, index 0~9가 class identity

## 성공 기준
- `dataset[0]` 호출 시 shape / overlap / bound 모두 통과
- Euclidean flow로 `b0 -> b1` trajectory GIF 생성
- Riemannian flow와 Euclidean flow trajectory 시각적 비교 가능
