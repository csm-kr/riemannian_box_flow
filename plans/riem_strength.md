# plans/riem_strength.md — Riemannian이 이기는 실험 셋업 탐색 (005~010)

## 배경
Phase 2.5 결과 (`plans/setup_analysis.md`, `outputs/comparison_4way/REPORT.md`):
- 본 toy(MNIST 10-box)에서 모든 metric (box-IoU, chart-MSE, signal-MSE)에서 **S-E (signal/eucl) 1위, C-E 4위**
- chart space의 강점(scale invariance, log multiplicative)이 발현되지 않음 — dataset의 size 분포가 좁음(median ratio ~2×)
- 가설: 적절한 셋업이면 Riemannian/chart space가 이길 수 있는 시나리오가 존재

본 plan은 chart space 특성을 적극 활용해 **Riem이 이기는 실험**을 6개(005~010) 제안.

## 핵심 통찰

| chart 공간의 이론적 강점 | 발현 조건 |
|---|---|
| Multiplicative size dynamics | box size가 orders of magnitude로 변할 때 |
| Scale invariance | 같은 dynamics가 모든 size에 적용 |
| Constant u target (C-R) | low-K inference, 학습 안정성 |
| Log-space gradient | 매우 작은 box에서도 잘 정의됨 |

| signal 공간의 강점 (현재 우위 원인) | |
|---|---|
| affine box ↔ signal | box-IoU 평가에 직접 유리 |
| decode 시 오차 선형 | exp 증폭 없음 |

→ chart의 강점이 signal의 강점을 능가하는 시나리오가 핵심.

---

## 실험 매트릭스 (005~010)

| ID | 이름 | 변경 | 학습 필요 | 가설 |
|----|----|----|----|----|
| **005** | Wide-scale dataset | 데이터: box size 50× 변화 | ✓ S-E + C-R (2개) | chart가 size에서 우위, IoU 격차 줄거나 역전 |
| **006** | Scale-fair metric eval | 평가: per-bucket weighted IoU + log-IoU | × | chart가 scale-fair metric에서 우위 |
| **007** | Hybrid model (signal-pos + chart-size) | 모델: 위치는 signal, 크기는 chart | ✓ (1개) | signal pos + chart size dynamics — best of both |
| **008** | Box-loss-trained chart | C-R 학습 loss를 box-space로 | ✓ (1개) | chart 모델이 box metric으로 직접 최적화되면 따라잡음 |
| **009** | K=1 single-shot | inference K=1 (one-step) | × | C-R의 constant u가 K=1에서 가장 정확 |
| **010** | Scale-aware Local chart | model.md §7.3 (body-frame chart) | ✓ (1개) | 진짜 Riemannian (parallel transport) — 모든 scale 동일 dynamics |

학습 필요 5개 (005에서 2개 + 007 + 008 + 010 = 5 학습) → 35k step × 5 ≈ 2.5시간.

---

## 실험별 상세 설계

### 005 — Wide-scale dataset
**문제**: 현재 dataset의 box size가 너무 좁음 (`w·h ∈ [0.004, 0.07]`, 거의 small bucket).
**변경**: `dataset/sampler.py`의 `sample_gt_boxes`에서 size를 더 넓게 (예: side `∈ [0.04, 0.6]`, area 200× 변화). 10개 box overlap 제약 그대로.
**산출**: `outputs/005_wide_signal/` (S-E, wide), `outputs/006_wide_chart/` (C-R, wide).
**평가**: 4-way comparison + size-stratified IoU. 가설: large bucket에서 chart 우위.

> **주의**: 새 dataset은 기존 ckpt와 호환 안 됨. 새 학습 필수. 또 데이터 split 구분 필요 (e.g., dataset class에 `wide=True` 옵션).

### 006 — Scale-fair metric (no new training)
**평가만**: 005 model + 4-way 모델 모두에 대해:
- **per-bucket-weighted IoU**: small/medium/large bucket 평균 → 균등 가중 (현재 mean IoU는 box 수에 비례)
- **log-IoU**: log(area_pred) vs log(area_gt) 비교 (size 정확도)
- **scale-relative center error**: `|c_pred - c_gt| / w_gt` (작은 box도 동등하게 취급)

가설: chart 모델이 scale-fair metric에서 더 잘 보임.

### 007 — Hybrid model (signal-pos + chart-size)
**모델**: `model/flow_hybrid.py` — input/output dim 4, 첫 2 채널은 signal-position, 마지막 2 채널은 chart-size.
- forward: backbone 그대로, output을 split해서 해석 변경
- fm_loss: position은 signal trajectory + constant u, size는 chart trajectory + constant u
- sample: position은 signal ODE, size는 chart ODE (둘 다 affine + exp decode)

**가설**: position 정확도(S-E 수준) + size 정확도(C-R 수준)을 동시에 — best of both.

### 008 — Box-loss-trained chart model
**변경**: `ChartNativeFlowModel.fm_loss`에서 chart-space MSE 대신 **box-space endpoint MSE**:
```python
c_t = (1-t)*c_0 + t*c_1
u_pred = model(c_t, t, image)
c_1_hat = c_t + (1-t)*u_pred             # straight-line endpoint estimate
b_1_hat = chart_decode(c_1_hat)
loss = MSE(b_1_hat, b_1)                  # box space loss!
```

**가설**: chart model이 box-space에 직접 정렬되면 IoU 격차 감소. exp() 증폭이 학습 신호로 들어감.

### 009 — K=1 single-shot inference (no new training)
**평가만**: 4 baseline의 ckpt를 K=1로 추론.
- C-R: `c_1_hat = c_0 + 1·u_pred = c_0 + (c_1 - c_0)` — u target이 exactly y_1-y_0면 정확
- S-E: 마찬가지, signal 공간에서 정확

**가설**: K=1에서 두 baseline 격차 가장 작거나 reverse — constant u 모델 (S-E, C-R)이 K=1에서 제일 잘함.

### 010 — Scale-aware Local chart (`model.md §7.3`)
**모델**: `model/flow_local.py` — chart가 매 step `b_t` 기준 reframe.
- input state: `ψ_{b_t}(b) = ((cx-cx_t)/w_t, (cy-cy_t)/h_t, log(w/w_t), log(h/h_t))`
- u_target: 위치 `(c_1^x - c_t^x)/((1-t)·w_t)`, 크기 `log(w_1/w_t)/(1-t)` (t→1 clamp)
- ODE: 매 step chart 재계산, body-frame integration

**가설**: 진짜 Riemannian (parallel transport on box manifold) — scale invariance가 가장 강하게 발현. 모든 size에서 동일한 dynamics.

---

## 작업 순서 (의존성)

1. **005** dataset 변경 + 2 학습 (signal + chart) — 가장 큼
2. **006** scale-fair metric (005 + 기존 4 모델 활용, 학습 없음)
3. **007** hybrid model + 학습
4. **008** box-loss chart + 학습
5. **009** K=1 inference (학습 없음, 기존 4 + 신규 모두에 적용)
6. **010** Local chart + 학습 (가장 어려움, 마지막)

병렬 학습 가능: GPU 96GB → 동시에 2~3개 학습 OK.

---

## 결정사항 / TBD

### 확정
- ✅ 모든 실험에서 학습 hyperparam 동일 (35k step, batch 64, hidden 256, depth 6)
- ✅ 평가는 box-space mean IoU + chart-MSE + scale-fair metric (006)
- ✅ 같은 init seed로 paired 비교

### TBD
- 005의 wide dataset이 학습 가능한지 (overlap 제약 + 큰 size 동시에)
- 010의 t→1 clamp 처리 (현재는 `t ∈ [0, 1-ε]`로 잘라야)
- hybrid 모델이 signal/chart 각각 학습보다 정말 합쳐서 잘 되는지 (capacity 분할 위험)

---

## 다음 단계 전환 조건 (Phase 2.5 → 마무리)
- 005~010 결과 → `outputs/comparison_riem_strength/REPORT.md`
- "Riem이 이기는 셋업"이 적어도 1개 발견 (예: 005 large bucket, 또는 010 scale-aware)
- 새 방법 정리: 본 toy를 넘어 일반화 가능한 셋업 권고
