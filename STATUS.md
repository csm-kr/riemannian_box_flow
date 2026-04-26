# STATUS.md

## 현재 위치
- Phase: **Phase 1 학습/추론 — Euclidean trajectory GIF 시각화부터 시작**

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
- [ ] `training/losses.py` (box L1 + GIoU; FM MSE는 flow_signal에 이미 있음)

## 최근 완료
- [x] `plans/training.md` Phase 1 학습/추론 계획 정리
- [x] **Phase 1 model 구성요소 완성** (master merge): signal chart, time_embed, rope2d, image_encoder, dit_block, backbone, flow_signal
- [x] `training/visualize.py` — ODE trajectory → frames → GIF 저장 (sanity 통과: 17 frames, untrained model)
