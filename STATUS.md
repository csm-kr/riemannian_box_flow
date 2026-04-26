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
- [ ] full-train run (50k step, batch 64) + train/val 곡선 + 후반 GIF 합리성 확인

## 최근 완료
- [x] `plans/training.md` Phase 1 학습/추론 계획 정리
- [x] **Phase 1 model 구성요소 완성** (master merge): signal chart, time_embed, rope2d, image_encoder, dit_block, backbone, flow_signal
- [x] `training/visualize.py` — ODE trajectory → frames → GIF 저장
- [x] `training/{config,trainer,train}.py` — Phase 1 trainer (FM-only, AdamW + warmup→cosine)
- [x] **Phase 1 short run 완료** (1000 step, batch 32, hidden 256/depth 6, DINOv2 frozen)
  - 71.7s 소요 (~14 step/s, RTX 6000)
  - train loss 4.53 → 1.23, val loss 1.47 → 1.17
  - GIF / ckpt → `outputs/short_run/`
- [x] Docker shm_size 8gb로 docker-compose.yml 업데이트 (DataLoader 워커 안정성)
