# STATUS.md

## 현재 위치
- Phase: **Phase 2.6 — Riem 이기는 셋업 탐색 완료** (`plans/riem_strength.md`, `outputs/comparison_riem_strength/REPORT.md`)
- 결론:
  - 6 시나리오 (wide ds, hybrid, box-loss, K sweep, scale-fair, local chart) 모두 학습/평가 완료
  - **box-IoU에서는 모두 S-E 우위** (wide 데이터에서도 격차 더 큼)
  - **scale-relative center error에서 hybrid가 S-E 미세 우위** — 새 방법 1개 발견
  - **새 방법: Hybrid (signal-pos + chart-size)** — `model/flow_hybrid.py`

## 완료
- [x] Docker 환경 (Dockerfile + shm_size 8gb)
- [x] 프로젝트 구조 정의
- [x] dataset/ 모든 모듈 구현
- [x] `plans/dataset.md` archive
- [x] `plans/model.md` Phase 1 + Phase 2 chart spec (§7.4 — `psi(b)=(cx,cy,log w,log h)`)
- [x] `plans/active.md` Phase 2 방향 정리 (사용자 spec 반영)
- [x] `plans/visualization.md` per-index / 비교 GIF 설계
- [x] **Phase 1 Euclidean 학습 완료**: 35k step (37k에서 shm 누수 종료, 패치 후 final.pt 확보)
  - val_loss 곡선: 1.34 (500) → 0.235 (5k) → 0.131 (20k) → 0.093 (30k) → 0.0835 최저 (35.5k) → plateau
  - ckpt: `outputs/001_fullrun/ckpt/final.pt` (= step_035000)
- [x] outputs 자동 numbering + TensorBoard 통합 + TB 6006 띄우기
- [x] shm 누수 패치 (`val_loader num_workers=0`, `train_loader persistent_workers=True`) — ISSUES.md 참조

## 진행 중
없음 — exp 013 (logit_native default init) 완료, **default init에서 001을 이김** (`outputs/comparison_default_logit/`).

## 최근 완료 (exp 013 — logit_native default-init)
- [x] 013_logit_native_default 학습 (35k step, val 0.075)
- [x] **001 vs 013 paired 비교 (5000 sample)**:
  - 001 signal:       0.8070 ± 0.102
  - **013 logit_native: 0.8150 ± 0.097 (+0.008, Wilcoxon p=0.008)**
- [x] 모든 GT-size bucket에서 013 우위 (small +0.015 가장 큼)
- [x] center_err 013 −0.004 우위, size_err 무승부 → logit pos가 학습 신호 충분히 줘서 signal과 동등 이상
- [x] 1~9 중 1위 등극 (013 = 0.815, 이전 best 001 = 0.813)
- [x] **결론**: logit chart는 signal chart보다 본질적으로 강함. 단, chart_native(003, 0.744) 처참한 점 보면 "multiplicative dynamics" 자체가 답 아니라 **"4성분 모두 symmetric logit + sigmoid clamp"** 가 답.

## 최근 완료 (exp 010/011/012 — small-init 비교)
- [x] init prior `small_size` (`w, h ~ U[0.01, 0.05]`, pos default) 추가 — `model/trajectory.py:sample_init_box`
- [x] 010_smallinit_signal (val 0.054)
- [x] 011_smallinit_chart_native (val 0.018, 그러나 box-IoU에서 최하위)
- [x] **012_smallinit_logit_native** — 4성분 모두 logit chart, `model/flow_logit_native.py` (val 0.069)
- [x] **paired 비교 결과 (5000 sample, mean IoU)**:
  - 🥇 **012 logit_native: 0.7966** (1st in all GT-size buckets, both center_err & size_err best)
  - 🥈 010 signal:           0.7755
  - ❌ 011 chart_native:      0.7109
- [x] **결론**: symmetric logit chart가 best — 010의 위치 정확도 + 011의 size 정확도를 동시에 가짐. logit(w) ≈ log(w) for small w 이므로 multiplicative 이득 유지하면서, logit(pos)이 [-3,+3] range 까지 확장되어 위치 학습 신호도 충분.

## 최근 완료 (Phase 2 검증)
- [x] Riemannian 35k step 학습 — `outputs/002_riemannian/` (val_loss 최저 0.0951)
- [x] **검증 5축 모두 완료** — `outputs/comparison/REPORT.md`:
  - 축 1: Eucl mean IoU 0.811 vs Riem 0.778 (Wilcoxon p≈0)
  - 축 1b: 차이 핵심은 size error (Riem 51% 더 큼); 위치는 거의 동률
  - 축 2: 모든 GT-size bucket에서 Eucl 우위 (small bucket 격차 가장 큼)
  - 축 2b: size-change ratio 클수록 Riem 격차 줄어듦 (부분 지지)
  - 축 3 K sweep: K=2 격차 0.101 → K=32 격차 0.009 (가설 정반대)
  - 축 4 seed variance: 동일 std 0.097 (p=0.65, 차이 없음)
  - 축 5 비교 GIF + per-index GIF
  - figures: size_iou.png, k_sweep.png, seed_variance.png

## 최근 완료 (Phase 2)
- [x] `model/flow_chart.py` — `ChartFlowModel(SignalFlowModel)`, `fm_loss`만 riemannian_trajectory 사용 (sanity 5케이스 통과)
- [x] trainer/train CLI에 `--model {signal,chart}` 추가 — 양쪽 8-step smoke 통과
- [x] `model/flow_signal.py` refactor — `fm_loss`가 `euclidean_trajectory` 호출, OLD vs NEW max abs diff 5e-7 (1-ULP 수준)
- [x] `model/trajectory.py` — signal_encode/decode + chart_encode/decode (psi, eps=1e-3) + euclidean/riemannian trajectory; 분석 u_t (위치는 Euclidean과 동일, size는 `6 w_t log(w1/w0)`), float64 numerical diff과 6e-7 이내 일치
