# active.md — 현재 전체 진행 방향

## 현재 단계
**Training Phase 1 — Euclidean(signal) full train** → `plans/training.md`

직전 단계 결과:
- Phase 1 model 7개 모듈 구현 완료 (master merge)
- Short run 1000 step: train loss 4.53→1.23, val 1.47→1.17 (sanity 통과)

대기 중 (다음 세션 직진):
- `outputs/{NNN:03d}_{run_name}/` 자동 numbering 도입
- TensorBoard writer trainer 통합
- Docker container 재시작 (`shm_size: 8gb` 적용)
- TB 백그라운드 실행 (port 6006)
- Full train 50k step 실행

## 전체 흐름
1. ~~Dataset~~ ✓
2. ~~Model (Phase 1)~~ ✓
3. **Training (Phase 1 Euclidean)** ← 현재
4. Visualization (K 비교 inference / GIF 비교)
5. Phase 2 — Riemannian (global / local chart)

## 다음 단계 전환 조건 (Visualization phase로)
- Phase 1 full train (50k step) 완료, train/val loss 수렴
- 후반 GIF에서 predicted box가 GT box에 합리적으로 수렴
- `plans/training.md` → `plans/archive/`로 이동
