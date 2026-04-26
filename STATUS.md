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
- [ ] Phase 1 학습 sanity check (1-step training loss 감소, overfit micro-set)

## 최근 완료
- [x] `model/charts/signal.py` (`box_to_signal`, `signal_to_box`) — master merge
- [x] `plans/training.md` Phase 1 학습/추론 계획 정리
- [x] `model/components/time_embed.py` (sinusoidal + 2-layer MLP) — master merge
- [x] **Phase 1 model 구성요소 완성** (`feature/model-phase1` 브랜치):
  - `model/components/rope2d.py` (2D RoPE, h/w split, complex 회전)
  - `model/components/image_encoder.py` (DINOv2 ViT-S/14 wrapper, frozen)
  - `model/components/dit_block.py` (self-attn + cross-attn(RoPE) + adaLN)
  - `model/backbone.py` (DINOv2 → adapter → DiT stack → adaLN-final → linear)
  - `model/flow_signal.py` (`SignalFlowModel`: forward / fm_loss / ODE Euler sample)
