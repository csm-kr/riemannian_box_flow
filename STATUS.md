# STATUS.md

## 현재 위치
- Phase: **Phase 3.0 — Logit-space FM 두 강점 검증 (시작)** (`plans/logit_strengths.md`)
- 비교 대상: 001 signal (final.pt) vs 013 logit_native (final.pt) — paired 비교 결과 +0.008 (013 우위, p=0.008)
- 검증할 두 가설:
  - **H1 boundary 자연 만족**: out-of-canvas rate / K-robustness (signal 은 ODE 끝점이 boundary 밖으로 튈 수 있음, logit 은 sigmoid 가 자동 보장)
  - **H2 scale-sensitive size**: fine size bucket (5) + ratio bucket (5) — `dlogit/dw = 1/w(1−w)` 가 작은 w 에 큰 gradient
- 새 학습 없음, 분석 코드 2개 작성 (`inference/boundary_audit.py`, `inference/size_dynamics.py`)

## 직전 단계 (Phase 2.6 — Riem 이기는 셋업 탐색 완료)
- `plans/riem_strength.md`, `outputs/comparison_riem_strength/REPORT.md`
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
없음 — **Phase 3.0 두 강점 검증 완료**. REPORT.md 종합 결론 작성됨.

## 최근 완료 (Phase 3.0 H2 — size_dynamics, 5000 sample × K=10)
- [x] `inference/size_dynamics.py` — 5 size × 5 ratio bucket × {iou, size_err, center_err}
- [x] `inference/size_dynamics_figures.py` — 4 figures
- [x] **H2a 지지**: small bucket IoU +0.0169 가장 큼 (mid +0.0091, large +0.0040, 단조 감소)
- [x] **H2b 지지**: ratio 4–8 격차 +0.0155 (가장 큼), ≤1.5 +0.0046, 1.5–2 +0.0016
- [x] **H2c 부분 지지**: size_err 모든 populated bucket 에서 logit 우위 (절대 격차는 작음 ≤5%)
- [x] **확장 발견**: IoU 격차의 주 원인은 **center_err 감소** (small bucket −13%) — logit 의 `1/(x(1−x))` 가 4성분 모두 boundary-amplifying
- [x] tiny/huge bucket 은 MNIST 에서 0개 — Opt-A wide-dataset 으로 보강 가능
- [x] `outputs/logit_strengths/{size_dynamics/, REPORT.md}` 종합 결론 포함

## 최근 완료 (Phase 3.0 H1 — boundary_audit, 5000 sample × K=2,4,8,16,32)
- [x] `inference/boundary_audit.py` — raw ODE Euler (clamp 우회), coord_oob/canvas_oob/excess_l1/IoU metrics
- [x] `inference/boundary_figures.py` — 3 figures (iou_vs_K, ooc_vs_K, excess_hist)
- [x] **K=2 에서 logit 우위 +0.048 IoU** (signal 0.7528 → logit 0.8008), K↑ 일수록 +0.001 ~ +0.011 로 수렴 — **K-robustness 가 H1 의 핵심 강점**
- [x] coord_oob_rate / excess_l1 모두 양쪽 모델에서 **0** — trained signal 도 [-3, 3] 자발적으로 지킴 → H1a / H1c 는 trivial 만족, differentiator 아님 (REPORT.md 에 정직하게 기록)
- [x] canvas_oob_rate: logit 모든 K 에서 0.4 ~ 0.8%p 낮음 (signal 0.79 ~ 1.49 %, logit 0.59 ~ 0.96 %)
- [x] paired Wilcoxon p ≪ 0.05 모든 K (K=2 p≈0, K=4 p=2.7e-128, …)
- [x] `outputs/logit_strengths/{boundary_audit/, REPORT.md}`

## 최근 완료 (exp 014 — corner_logit default-init)
- [x] 014_corner_logit_default 학습 (35k step, 최저 val 0.279 @ 31k, final val 0.372)
  - chart: `y = (logit((cx−w/2)/(1−w)), logit((cy−h/2)/(1−h)), logit w, logit h)` — left/top corner 정규화 + logit
  - decode `cx = w/2 + (1−w)·sigmoid(y_0)` 로 박스 자동 in-canvas
- [x] **001 vs 014 paired 비교 (5000 sample)**:
  - 001 signal:        0.8075 ± 0.102
  - 014 corner_logit:  **0.7487 ± 0.106 (−0.059, Wilcoxon p≈0)**
  - 모든 GT-size bucket에서 001 우위 (small −0.080 가장 큼)
  - center_err / size_err 모두 014가 0.006 더 큼
- [x] **결론**: in-canvas 자동 보장 property는 IoU 우위를 만들지 않음. corner-logit에서 pos가 size에 의해 정규화 (1−w 분모) → pos·size가 chart에서 coupled. 013 logit_native는 4성분 완전 독립이라 우승 (0.815). **"4성분 symmetric + 독립"이 핵심**이라는 가설 재확인.

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
