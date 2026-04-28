# plans/setup_analysis.md — 좌표계 셋업 비교 분석

## 배경
Phase 2 검증에서 본 repo는 Euclidean이 모든 axis에서 우위였음 (`outputs/comparison/REPORT.md`).
그러나 외부 참조 repo [`csm-kr/riemannian_flow_det`](https://github.com/csm-kr/riemannian_flow_det)는 정확히 같은 도메인(box flow detection)에서 **Riemannian이 우위**라 보고함:
- Tail loss 0.028 vs 0.056 (Eucl baseline)
- Max error 16.9px vs 49.1px (3× 개선)
- 학습 안정성 std 70× / p99 48× 차이

→ **두 실험 결과가 정반대.** 본 분석은 그 차이의 원인을 규명하고, 본 repo에 어떤 추가 실험이 필요한지 정리한다.

---

## 1. 핵심 차이 — 모델 state space

| 측면 | 본 repo | csm-kr/riemannian_flow_det |
|---|---|---|
| **Model 입출력 state** | **signal** `s = 6b - 3` | **chart** `y = (c_x, c_y, log w, log h)` |
| **Prior `s_0`** | clip(N(0,I)) in signal | N(0,I) in chart |
| **Eucl trajectory** | signal-state 직선<br>`s_t = (1-t)s_0 + t s_1`<br>**u = s_1 − s_0 (상수)** | cxcywh-box 직선 → chart로 변환<br>**u_chart ∝ 1/w_t (state-dep, 발산)** |
| **Riem trajectory** | chart-state 직선 → signal로 변환<br>**u_signal = 6 w_t log(w_1/w_0) (state-dep)** | chart-state 직선 그대로<br>**u = b_1 − b_0 (상수)** |
| **결과** | Eucl 우위 | Riem 우위 |

---

## 2. 핵심 패턴 (가설)

> **"모델 학습 state와 trajectory 직선이 그어진 state가 일치하는 쪽이 항상 이긴다."**

이유:
1. 두 state가 일치하면 target velocity가 **상수** → 학습 신호 강함, optimizer 쉽게 수렴
2. ODE Euler 적분이 상수 velocity에 대해 **모든 K에서 정확** (truncation error 없음)
3. State 불일치 baseline은 변환에서 state-dependent u가 발생 → 학습+적분 둘 다 손해

본 repo와 그쪽 repo의 결과는 모순이 아니라 같은 원리의 양면.

---

## 3. 본 repo의 비대칭

본 repo에서:
- Eucl model + Eucl trajectory: state 일치, u 상수 (best case)
- Eucl model + Riem trajectory: state 불일치, u state-dep (worst case)

따라서 **"Riemannian이 좋다/나쁘다"의 결론을 내리려면 같은 비교를 chart space model에서도 해야 한다.**

---

## 4. 실험 매트릭스 (제안)

(model state, trajectory) 4 조합:

| ID | Model state | Trajectory | u 형태 | 예상 |
|---|---|---|---|---|
| **S-E** ✓ | signal | signal-straight | 상수 `s_1−s_0` | best (현 Eucl baseline) |
| **S-R** ✓ | signal | chart-straight | state-dep `6 w_t log(...)` | worst (현 Riem baseline) |
| **C-E** ✗ | chart | cxcywh-straight | state-dep `≈ 1/w_t` | worst (그쪽 Eucl baseline) |
| **C-R** ✗ | chart | chart-straight | 상수 `b_1−b_0` | best (그쪽 Riem baseline) |

✓ = 본 repo에 이미 학습 완료. ✗ = 추가 학습 필요.

가설이 맞다면:
- **S-E ≈ C-R** (둘 다 best case, mean IoU 비슷할 것)
- **S-R ≈ C-E** (둘 다 worst case)
- best/worst 격차는 chart 종류와 무관, **state 일치 여부**가 결정

이를 검증하면 "Riem vs Eucl"이 아니라 **"state 일치가 핵심"**이 본 repo의 결론이 됨.

---

## 5. 작업 순서

### 5.1 코드 변경
새 모델 클래스가 필요. backbone은 동일(DiT)이지만 state 공간이 다름:

```
model/
├─ flow_signal.py   ← 기존 (S-E baseline)
├─ flow_chart.py    ← 기존 (S-R baseline; signal model + Riem trajectory)
└─ flow_chart_native.py  (신규)
   ├─ 입력 state: chart y_t = (c_x, c_y, log w, log h)
   ├─ 출력: chart velocity v_y
   ├─ Prior: y_0 ~ N(0, I) (또는 분포 조정 필요 — TBD)
   ├─ Eucl-in-chart trajectory (C-E): cxcywh 직선 → chart 변환, target = state-dep
   └─ Riem-in-chart trajectory (C-R): chart 직선 → target = b_1 − b_0 (상수)
```

또는 단일 클래스에 `coord` 파라미터로 분기.

### 5.2 Prior 분포 정렬 (TBD)
- 본 repo의 signal space prior `clip(N(0,I), -3, 3)`은 box space에서 거의 균등 분포
- chart space prior `N(0,I)`은 box space에서 분포가 다름:
  - `c_x, c_y ~ N(0,1)` → 대부분 [0,1] 밖. 클램프 또는 분포 조정 필요.
  - `log w, log h ~ N(0,1)` → `w ~ LogNormal(0,1)` (median 1, 99%까지 ~10) — 대부분 [0,1] 밖.
- 그쪽 repo는 어떻게 처리하는지 확인 필요 (아마 box를 [0,1]로 정규화하지 않거나, 다른 prior 정의).
- 우리 데이터(box ∈ [0,1]^4)에 맞는 chart prior는 **별도 설계** 필요.

### 5.3 학습
- C-R 모델 35k step 학습 → `outputs/{NNN}_chart_riem/`
- C-E 모델 35k step 학습 → `outputs/{NNN}_chart_eucl/`
- 시간: 각 ~30분 → 총 1시간

### 5.4 비교
- 4 baseline (S-E, S-R, C-E, C-R)을 같은 metric으로 비교
- mean IoU, size-stratified, K sweep 모두
- 가설 검증: 정말 state-매칭이 결정적인가?

---

## 6. 주의사항 / 회의적 관찰

1. **Prior 차이가 결과를 흐릴 수 있음.** chart space에서 `N(0,I)` 시작은 box space에서 매우 다른 분포 — 학습 난이도 자체가 달라질 수 있음. 통제 필요.
2. **그쪽 repo의 "Eucl baseline"이 사실 의도적으로 약한 셋업**일 수 있음. cxcywh 직선 → chart 변환은 자연스러운 비교가 아님. 진짜 공정한 Eucl은 state space에 맞춘 직선 (=우리 Eucl).
3. 따라서 "본 repo Eucl(S-E) vs 그쪽 Riem(C-R)" 비교가 진짜 공정 비교일 수 있음.
   - 둘 다 best case (state 일치 + 상수 u)
   - 그러면 차이는 chart 종류 (signal vs chart) — multiplicative dynamics가 정말 필요한지의 질문

---

## 7. 결정 필요

- [ ] **옵션 A**: C-R만 학습 (1개 baseline 추가) → S-E vs C-R 비교가 진짜 공정 비교. 1시간.
- [ ] **옵션 B**: 4 baseline 모두 학습 → 가설 검증 완전. 1시간.
- [ ] **옵션 C**: 코드 변경 없이 분석/리포트만. 본 repo 결론은 유지하되, 그쪽 repo와의 차이를 explain.

추천: **옵션 A**. C-E는 의도적으로 약한 셋업이라 학습 가치 낮음. 진짜 궁금한 건 "S-E (signal+const) vs C-R (chart+const)" — 둘 다 best case인데 chart 종류가 IoU에 차이를 만드는가?

---

## 8. 다음 단계 전환 조건
- 옵션 결정 → C-R 모델/학습 완료
- C-R vs S-E 비교 → REPORT.md에 새 섹션 추가
- 결론: Riemannian이 size 변화에 진짜 도움이 되는지 (chart 효과만 격리)

---

## 9. 산출물
```
plans/setup_analysis.md       (이 문서)
model/flow_chart_native.py    (옵션 A/B에서 추가)
outputs/{NNN}_chart_riem/     (옵션 A/B에서 학습 결과)
outputs/comparison/REPORT.md  (4-way 비교 추가)
```
