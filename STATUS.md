# STATUS.md

## 현재 위치
- Phase: **Dataset 구현 완료, 실행 검증 필요**

## 완료
- [x] Docker 환경 (Dockerfile, docker-compose.yml)
- [x] 프로젝트 구조 정의
- [x] dataset/ 모든 모듈 구현
  - [x] box_utils.py
  - [x] sampler.py
  - [x] mnist_source.py
  - [x] canvas.py
  - [x] visualize.py
  - [x] mnist_box_dataset.py

## 진행 중
- [ ] 컨테이너에서 `python -m dataset.mnist_box_dataset` 실행 및 cv2.imshow 확인
