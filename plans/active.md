# active.md — 현재 전체 진행 방향

## 현재 단계
**Phase 3.0 완료** — Logit-space FM 두 강점 검증 끝. plan 은 `plans/archive/logit_strengths.md`, 결론은 `outputs/logit_strengths/REPORT.md`.

다음 진행 옵션:
- (Opt-A) 013 wide-dataset 학습 → tiny / huge bucket 검증
- (Future) COCO 등 실제 도메인 H2 재현
- (마무리) PR / README 통합 후 다음 phase 결정

---

## 직전 단계 (Phase 3.0 — Logit-space FM 두 강점 검증 완료)

`plans/archive/logit_strengths.md`, `outputs/logit_strengths/REPORT.md` 참조.
- **H1 (boundary)** — K=2 IoU 격차 +0.048 (signal 0.7528 → logit 0.8008), K↑ 일수록 +0.001 ~ +0.011 로 수렴. **K-robustness 가 본 강점**.
- **H2 (scale-sensitive size)** — small bucket +0.017, ratio 4–8 +0.016. **center_err 가 IoU 우위의 주된 원인** (size_err 도 logit 우위지만 절대 격차는 작음). logit 의 `1/(x(1−x))` 가 4성분 모두 boundary-amplifying.

## 직전 단계 (Phase 2.6 — Riem 이기는 셋업 탐색 완료)

`plans/archive/riem_strength.md`, `outputs/comparison_riem_strength/REPORT.md` 참조.
- 6 시나리오 모두 box-IoU 에서 S-E 우위. hybrid 만 scale-relative center error 미세 우위.
- **새 방법**: hybrid (`model/flow_hybrid.py`) — 향후 도메인 검증 대상.

## 직전 단계 (Phase 2.5 — 좌표계 셋업 분석 완료)

`plans/archive/setup_analysis.md` 참조.
- Phase 2 검증에서 Eucl이 모든 axis 우위였으나, 외부 repo (`csm-kr/riemannian_flow_det`)는 같은 도메인에서 Riem 우위 보고. 분석 결과:
- 두 repo의 차이는 **모델 state space** (signal vs chart) 선택
- 우리 Riem = signal model + chart trajectory (state-dep u, 학습 어려움)
- 그쪽 Riem = chart model + chart trajectory (constant u, 학습 쉬움)
- → **"Riem이 좋다/나쁘다"가 아니라 "model state ↔ trajectory state 일치 여부"가 결정**

## 직전 단계 (Phase 2 완료)
**Phase 2 — Riemannian baseline 구축 → Euclidean과 공정 비교**

직전 결과 (Phase 1 마무리):
- Euclidean (signal) 35k step 학습 완료, val_loss 0.0835 최저 → `outputs/001_fullrun/ckpt/final.pt`
- shm 누수 패치 적용 (`val_loader num_workers=0`, `train_loader persistent_workers=True`)
- default `total_steps`도 35k로 조정

---

## A. Phase 2 핵심 아이디어 (사용자 spec 반영)

### A-1. Chart 정의 (단순화)
```
psi(b)     = (cx, cy, log w, log h)        # 위치는 그대로, size만 log
psi_inv(y) = (y_cx, y_cy, exp(y_lw), exp(y_lh))
```
- `model.md §7`의 Global chart에서 위치 부분의 `3(2c-1)` scaling을 빼고 raw `[0,1]`로.
- 이유: 위치는 이미 [0,1]로 bounded — signal space로 굳이 옮길 필요 없음. **size만 multiplicative dynamics를 받기 위해 log 사용.**
- `model.md §7`의 Local (scale-aware) chart는 이번 Phase 2 범위에서 **제외** — 한 chart로만 비교.

### A-2. Euclidean vs Riemannian — **모델 / ODE는 동일, 학습 target만 다름**

```
공통:
  - 모델 출력 = signal-space velocity (B, 10, 4)
  - ODE inference = signal space에서 Euler 적분 후 signal_decode로 box 복원
  - backbone, query 수 (10), classification/Hungarian 없음 — 모두 동일

차이:
  Euclidean (학습 target):
    s_0 = signal_encode(b_0)
    s_1 = signal_encode(b_1)
    s_t = (1-t)*s_0 + t*s_1
    u   = s_1 - s_0                          # constant in t

  Riemannian (학습 target):
    y_0 = psi(b_0)
    y_1 = psi(b_1)
    y_t = (1-t)*y_0 + t*y_1                  # chart-space 직선
    b_t = psi_inv(y_t)
    s_t = signal_encode(b_t)
    u_t = d(s_t)/dt                           # state-dependent (size 성분이 b_t에 의존)
```

### A-3. 분석적 u_t (Riemannian) — 수치차분 필요 없음
`s = 6b - 3` (signal_encode), `psi_inv`로 b 복원 후 미분 chain rule:

| 성분 | u_t |
|------|-----|
| cx, cy | `6 * (cx_1 - cx_0)` = signal-space 차이와 동일 (즉 위치는 Euclidean과 같음) |
| w, h | `6 * w_t * (log w_1 - log w_0)` = `6 * w_t * log(w_1/w_0)` (state-dependent) |

→ **둘의 실질 차이는 size 성분의 학습 target뿐.**
→ 수치차분은 sanity check용으로만 (analytical과 거의 일치하는지 확인).

### A-4. 시각화 비교
- 같은 image, 같은 init seed에서 두 모델로 ODE inference → 같은 K=10 timesteps.
- 사용자 spec: `t = 0.0, 0.1, ..., 1.0` (11 frames, K=10)
- Per-index trajectory GIF (index별 별도 + optional 통합 GIF)
- 한 sample에서 Euclidean / Riemannian 두 trajectory를 가로로 합친 224×448 GIF (Signal | Riemannian)

---

## B. 결정사항 / 가정 (사용자 spec 기반 확정)

1. ✅ **Chart 1종만**: `psi(b) = (cx, cy, log w, log h)` (단순 Global, scale-aware Local 제외)
2. ✅ **모델 / 출력 / ODE는 Euclidean과 동일** — 학습 target만 다른 두 baseline
3. ✅ Riemannian u_t는 **분석적**으로 계산 (state-dependent size velocity); 수치차분은 sanity check
4. ✅ ODE inference 기본 K=10 (timesteps `0.0, 0.1, ..., 1.0`)
5. ✅ Backbone warm-start 안 함 — Riemannian도 from-scratch (공정 비교)
6. ✅ Phase 1과 동일한 35k step, batch 64, hidden 256, depth 6, DINOv2 frozen

## C. TBD (구현 시 결정)
- log w, log h 계산 시 `eps` 클램프 값 (`max(w, 1e-3)` 정도? signal init에서 b_0이 매우 작은 size를 만들 수 있음)
- visualization layout 옵션 (가로 panel만? overlay도?)

---

## D. 파일 구조 (Phase 2 추가)

기존 (Phase 1):
- `model/charts/signal.py` — signal encode/decode
- `model/flow_signal.py` — `SignalFlowModel`
- `training/trainer.py` — chart-agnostic trainer

추가/변경 (Phase 2):
- **새**: `model/charts/chart.py` — `psi(b)`, `psi_inv(y)` (eps clamp 포함)
- **새**: `model/trajectory.py` — `signal_encode/decode`, `chart_encode/decode`, `euclidean_trajectory`, `riemannian_trajectory` (사용자 spec의 `src/models/trajectory.py`)
- **refactor**: `model/flow_signal.py` — `fm_loss`가 `model/trajectory.euclidean_trajectory` 호출 (기존 동작 보존)
- **새**: `model/flow_chart.py` — `ChartFlowModel` (Riemannian baseline; backbone 공유, `fm_loss`만 다름)
- **확장**: `training/visualize.py` — Euclidean vs Riemannian 비교 GIF 생성기

> 사용자 spec의 `src/...` prefix는 본 repo가 top-level layout을 쓰므로 적용 안 함 (`dataset/`, `model/`, `training/`).

---

## E. 작업 순서 (TODO에 반영)

1. **`model/trajectory.py`** — chart/signal 함수 + 두 trajectory 함수 (TDD: shape, round-trip, 수치차분 vs 분석 일치)
2. **`model/charts/chart.py`** — `psi`, `psi_inv` (TDD: round-trip, eps 처리)
3. **`model/flow_signal.py` refactor** — `fm_loss`를 `euclidean_trajectory` 사용으로 (기존 sanity check 통과 유지)
4. **`model/flow_chart.py`** — `ChartFlowModel.fm_loss` (TDD: shape, t→0/t→1 한계, Euclidean과 위치 성분 일치)
5. **Riemannian 35k step 학습** — `outputs/{NNN}_riemannian/`
6. **비교 GIF / metric** — 같은 sample, 같은 init seed로 Euclidean (`001_fullrun/final.pt`) vs Riemannian → 가로 panel GIF + box-space mean IoU 표

---

## 전체 흐름
1. ~~Dataset~~ ✓
2. ~~Model (Phase 1)~~ ✓
3. ~~Training Phase 1 Euclidean~~ ✓ (35k step)
4. ~~Phase 2 — Riemannian baseline + 비교~~ ✓
5. ~~Phase 2.5 — 셋업 분석~~ ✓
6. ~~Phase 2.6 — Riem 이기는 셋업 탐색~~ ✓
7. ~~Phase 3.0 — Logit-space FM 두 강점 검증~~ ✓
8. **마무리: plans archive + README + PR ← 현재**

## 다음 단계 옵션
- (Opt-A) 013 wide-dataset 학습 → tiny / huge bucket 검증
- (Future) 실제 도메인 (COCO 등) 에서 H2 격차 재현
- (Future) hybrid 모델 (`model/flow_hybrid.py`) 다른 도메인 적용
