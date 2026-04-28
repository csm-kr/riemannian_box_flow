# plans/logit_strengths.md — Logit-space FM 두 강점 검증 (Phase 3.0)

## 한 줄 요약
"왜 logit chart (013) 가 signal chart (001) 보다 강한가?" 를 **두 가설로 분해**해서 정량 증거 확보.
새 학습 없이 이미 학습된 ckpts (001 signal, 013 logit_native) 만으로 분석 가능.

---

## 0. 배경 (사용자 spec 정리)

### 0.1 기존 box flow (signal-space FM)
- Gaussian noise → bounded signal `[−s, s]^4` → flow matching
- 학습은 bounded coordinate에서 이루어지지만 box의 boundary constraint·scale 특성 미반영
- `cx, cy, w, h` 모두 단순 Euclidean 변수처럼 다룸

### 0.2 Logit-space FM
- `b ∈ (0,1)^4` → `z = logit(b) ∈ R^4` → **unbounded** space에서 flow matching
- sigmoid decoding이 box range `[0,1]^4`을 자연스럽게 보장
- `w, h` 변화가 boundary-aware + scale-sensitive (작은 box / 큰 box 학습 불균형 완화)

### 0.3 두 강점
| # | 강점 | 작용 메커니즘 | 기대 효과 |
|---|------|-------------|---------|
| **H1** | boundary 자연 만족 | sigmoid decode → box 자동으로 `[0,1]^4` | invalid box 0%, ODE step 수 (K) 적어도 robust |
| **H2** | scale-sensitive size | `logit(w)` 의 도함수 = `1 / w(1−w)` → 작은 w 에서 gradient 가 커짐 | small box bucket / 큰 size-ratio 변환에서 학습 신호 충분 |

---

## 1. 검증 전략

### 1.1 비교 대상
| 약어 | 모델 | ckpt | 학습 |
|------|------|------|------|
| **S** | `SignalFlowModel` (signal chart) | `outputs/001_fullrun/ckpt/final.pt` | 35k step, default init |
| **L** | `LogitNativeFlowModel` (logit chart) | `outputs/013_logit_native_default/ckpt/final.pt` | 35k step, default init |

> 이미 paired 비교 결과: 001 = 0.8070, 013 = **0.8150** (+0.008, p=0.008). small bucket 격차 +0.015 가장 큼.
> Phase 3.0 은 이 차이를 **두 메커니즘 (H1, H2) 으로 분해**.

### 1.2 데이터 / seed
- val split (MNIST), 5000 sample, **paired** (compare.py 의 cache_val_batches 재활용)
- 같은 init seed 로 b_0 공유 → H1·H2 metric 모두 paired

---

## 2. H1 (boundary 자연 만족) — 실험 설계

### 2.1 핵심 metric (새로 정의)
1. **out-of-canvas rate** — `cx − w/2 < 0` ∨ `cx + w/2 > 1` ∨ `cy ± h/2` 범위 밖인 box 비율
2. **boundary excess (L1)** — clamp(0,1) 전 raw box 좌표가 `[0,1]` 밖으로 튀어나간 합 (per-box 평균)
3. **K-robustness** — `K ∈ {2, 4, 8, 16, 32}` 에서 IoU + out-of-canvas rate 곡선

> S 의 ODE Euler 는 signal space `[-3, 3]^4` 에서 적분 → 적분 step 이 적으면 끝점이 `[-3, 3]` 을 벗어남 → decode 후 `[0, 1]` 밖.
> L 은 logit space 에서 무제한 적분해도 sigmoid 가 항상 `[0, 1]^4` 보장 → out-of-canvas 0%.

### 2.2 가설
- **H1a**: out-of-canvas rate(S) ≫ out-of-canvas rate(L) ≈ 0
- **H1b**: K=2 / K=4 같은 적은 step 에서 S 의 IoU 격차가 가장 커지고, K↑ 일수록 격차 줄어듦
- **H1c**: 하지만 *clamp 후* IoU 차이의 절대량은 작을 수 있음 — clamp 가 일종의 boundary "치료" 역할 → 이 관찰 자체가 결론의 일부

### 2.3 구현 (`inference/boundary_audit.py` — 신규)
```python
# pseudocode
def boundary_audit(ckpt_S, ckpt_L, Ks=(2, 4, 8, 16, 32), out_dir):
    batches = cache_val_batches(...)        # compare.py 재활용
    rows = []
    for K in Ks:
        for ckpt in (ckpt_S, ckpt_L):
            pred_raw, init, gt = collect_predictions_RAW(...)   # NO clamp(0,1)
            ooc_rate = ((pred_raw - clamp(pred_raw, 0, 1)).abs() > 0).any(-1).float().mean()
            excess  = (pred_raw - clamp(pred_raw, 0, 1)).abs().sum(-1).mean()
            iou     = iou_xywh(clamp(pred_raw, 0, 1), gt).mean()
            rows.append((K, model_name, ooc_rate, excess, iou))
    save(rows → boundary_metrics.csv)
    plot(K, iou)              # K-robustness 곡선
    plot(K, ooc_rate)         # boundary excess 곡선
```

> ⚠️ `inference/compare.py:collect_predictions` 는 model.sample 안에서 `clamp(0,1)` 을 한다 (`flow_logit_native.py:78` 등).
> RAW prediction 이 필요하므로 **flag 추가 또는 별도 함수** 작성.

### 2.4 보고 figure
- `boundary_audit/iou_vs_K.png` — S vs L IoU vs K
- `boundary_audit/ooc_vs_K.png` — out-of-canvas rate vs K
- `boundary_audit/excess_hist.png` — clamp distance 분포 (K=10 기준)

---

## 3. H2 (scale-sensitive size dynamics) — 실험 설계

### 3.1 핵심 metric
이미 `inference/compare.py` 에 있는 size-bucket / ratio-bucket 분석을 더 fine-grained 하게.

1. **size bucket fine** — GT box area `√(w·h)` 를 5 buckets:
   - tiny  `[0.0,  0.05)`
   - small `[0.05, 0.10)`
   - mid   `[0.10, 0.20)`
   - large `[0.20, 0.40)`
   - huge  `[0.40, 1.00]`
2. **size-only error** — `size_err = |w_pred − w_gt| + |h_pred − h_gt|` (center 영향 제거)
3. **size_change_ratio bucket** — `r = max(w_1/w_0, w_0/w_1)` 5 buckets:
   - `[1, 1.5)`, `[1.5, 2)`, `[2, 4)`, `[4, 8)`, `[8, ∞)`

### 3.2 가설
- **H2a**: tiny / small bucket 에서 L 우위가 가장 크고, huge bucket 에서는 격차 사라짐 (또는 역전 가능)
- **H2b**: size_change_ratio ≥ 4 인 케이스에서 L 우위가 ratio 작은 케이스보다 큼
- **H2c**: size-only error 만 봐도 H2a·H2b 가 그대로 유지 (= center error 가 우위 원인이 아님)

### 3.3 구현 (`inference/size_dynamics.py` — 신규)
```python
# pseudocode
def size_dynamics(ckpt_S, ckpt_L, out_dir):
    batches = cache_val_batches(...)
    pred_S, _, gt = collect_predictions(model_S, batches, K=10, seed=0)
    pred_L, _, _  = collect_predictions(model_L, batches, K=10, seed=0)

    fine_buckets = stratify_by_size(gt, thresholds=[0.05, 0.10, 0.20, 0.40])
    ratio = size_change_ratio(init_box, gt)         # already in metrics.py
    ratio_buckets = stratify_by_ratio(...)

    for metric_name, metric_fn in [("iou", iou_xywh),
                                    ("size_err", size_error),
                                    ("center_err", center_error)]:
        write_csv(metric_fn(pred_S, gt), metric_fn(pred_L, gt),
                   fine_buckets, ratio_buckets, out_dir)
        plot_buckets(...)
```

`metrics.py` 의 기존 함수 재활용 + size bucket threshold 만 fine 으로 교체.

### 3.4 보고 figure
- `size_dynamics/iou_by_fine_size.png` — 5 size buckets bar chart, S vs L
- `size_dynamics/size_err_by_fine_size.png` — center 영향 제거한 비교
- `size_dynamics/iou_by_ratio.png` — size_change_ratio bucket
- `size_dynamics/iou_diff_heatmap.png` — (size bucket × ratio bucket) IoU diff (L − S)

---

## 4. 종합 REPORT (`outputs/logit_strengths/REPORT.md`)

```
# Logit-space FM vs Signal-space FM — 강점 분해 보고서

## 결론
- H1 (boundary): … (out-of-canvas rate, K-robustness 결과)
- H2 (scale-sensitive size): … (small-bucket 격차, ratio-bucket 격차)

## 1. H1 boundary
- table: K vs (IoU_S, IoU_L, ooc_rate_S, ooc_rate_L, excess_S, excess_L)
- figures: 3개

## 2. H2 scale-sensitive
- table: fine-bucket 5개, ratio-bucket 5개 — IoU + size_err
- figures: 4개

## 3. 한계
- ckpts 가 같은 step (35k) / 같은 hp 인지 명시
- val sample 5000 / paired
- 두 강점 모두 "logit chart" 의 단일 변경에서 비롯된 것이므로 분리 실험은 사실상 불가능 (sigmoid 가 둘 다 책임). → "메커니즘 두 면을 보여주는 두 metric 묶음"으로 해석.
```

---

## 5. 옵션 추가 실험 (필요 시)

| 약어 | 내용 | 판단 |
|------|------|------|
| Opt-A | 013 wide-dataset 학습 → 005/006/wide 에서 logit 강점 재확인 | size 분포 매우 넓을 때 격차 → H2 의 강한 증거 |
| Opt-B | 013 small-init prior 재학습 vs 010 (signal small-init) — 이미 있음 (012 = 0.7966 vs 010 = 0.7755) | H2 지지 증거로 REPORT 에 인용만 |
| Opt-C | logit chart 의 학습 동역학 logging — 학습 중 size dim 의 MSE 곡선만 isolate | H2 메커니즘 직접 확인 가능, 그러나 새 학습 필요 |

> Opt-A 만 우선 후보 (1× 35k 학습). Opt-B/C 는 결과 빈약 시 추가.

---

## 6. 파일 구조 (Phase 3.0 추가)

```
inference/
├─ boundary_audit.py      ← H1 metric + K sweep (raw prediction 채취)
└─ size_dynamics.py       ← H2 fine-bucket + ratio-bucket 분석

outputs/logit_strengths/
├─ boundary_audit/
│  ├─ boundary_metrics.csv
│  ├─ iou_vs_K.png
│  ├─ ooc_vs_K.png
│  └─ excess_hist.png
├─ size_dynamics/
│  ├─ stratified.csv
│  ├─ iou_by_fine_size.png
│  ├─ size_err_by_fine_size.png
│  ├─ iou_by_ratio.png
│  └─ iou_diff_heatmap.png
└─ REPORT.md
```

> `model/`, `training/` 변경 없음. 분석 전용 phase.

---

## 7. 작업 순서 (TODO 반영)

1. **`inference/boundary_audit.py` 작성** — `collect_predictions_RAW` (no clamp) + K sweep + 3 figures (TDD: same-ckpt smoke → ooc 동일, K-robustness 곡선 단조성)
2. **`outputs/logit_strengths/boundary_audit/` 결과 생성 + figures**
3. **`inference/size_dynamics.py` 작성** — fine-bucket + ratio-bucket + 4 figures (TDD: same-ckpt smoke → 모든 metric diff = 0)
4. **`outputs/logit_strengths/size_dynamics/` 결과 생성**
5. **종합 `REPORT.md` 작성** — H1 / H2 결론 + 한계
6. (옵션) Opt-A 013 wide-dataset 학습 → REPORT 에 부록

---

## 8. 결정사항
1. ✅ 새 모델 학습 없음 (옵션 Opt-A 만 별도 결정)
2. ✅ 두 강점은 같은 chart 변경에서 비롯되므로 "분리 학습" 대신 "분리된 metric 묶음" 으로 검증
3. ✅ K sweep 은 **raw prediction (clamp 전)** 도 같이 보고 — 두 모델의 차이가 clamp 에 가려지지 않게
4. ✅ paired comparison 유지 (같은 init seed, 같은 val batch cache)
5. ✅ size bucket / ratio bucket 모두 5-fine 으로

## 9. TBD
- out-of-canvas 의 "정도" metric: L1 excess vs L∞ excess 중 어느 쪽이 해석하기 좋은지 (구현 시 결정)
- size_change_ratio bucket threshold 의 정확한 cut-point — 분포 보고 quantile 로 자르는 것도 옵션

---

## 10. 다음 단계 전환 조건 (Phase 3.0 → 마무리)
- H1·H2 두 묶음 모두 figure + table + p-value 확보
- REPORT.md 결론에서 두 강점이 각각 데이터로 뒷받침
- (불일치 시) 어떤 메커니즘이 책임 큰지 — 예: ooc_rate 가 K=10 에서 거의 0 이면 H1 은 K↓ 영역에서만 유효함을 명시
