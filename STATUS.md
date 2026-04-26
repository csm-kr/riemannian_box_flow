# STATUS.md

## 현재 위치
- Phase: **Model Phase 1 — Euclidean(signal) flow 구현 시작**

## 완료
- [x] Docker 환경 (Dockerfile, docker-compose.yml)
- [x] 프로젝트 구조 정의
- [x] dataset/ 모든 모듈 구현
- [x] `python -m dataset.mnist_box_dataset` sanity check 통과
- [x] `plans/dataset.md` archive 이동
- [x] `plans/model.md` 설계 정리
  - Phase 1 = Euclidean(signal) 우선, Riemannian은 Phase 2
  - 학습 9-step 수식 / clip(N(0,I)) init / FM loss 확정
  - Riemannian (global/local) 설계는 §7에 reference로 보존

## 진행 중
- [ ] `model/components/time_embed.py` (sinusoidal + 2-layer MLP)

## 최근 완료
- [x] `model/charts/signal.py` (`box_to_signal`, `signal_to_box`) — sanity check 통과 (endpoint / round-trip / batched / dtype / autograd)
