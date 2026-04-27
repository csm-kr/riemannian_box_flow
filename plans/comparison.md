# plans/comparison.md — Phase 2 검증 (Euclidean vs Riemannian)

## 목표
"Riemannian이 Euclidean보다 더 좋다"를 **공정한 metric + 통계적 의미**로 검증.
단순히 비교 GIF "그럴듯해 보임"으로는 부족.

---

## 핵심 가설
`psi(b) = (cx, cy, log w, log h)` chart의 강점은 **size의 multiplicative dynamics**.
- 작은 box ↔ 큰 box로 이동 시 등속(linear)이 아니라 비율(log) 변화가 자연스러움.
- → 작은 box, 다양한 size 분포에서 Euclidean보다 우위 예상.

---

## 검증 4 축

### 축 1 — Box-space **mean IoU** (필수, primary)
- val split 전체 (N samples)
- 두 모델 모두 **같은 image, 같은 init seed**로 ODE → predicted box → `iou_xywh(pred, gt)`
- index-wise 평균 (10 query) → sample 평균 → val 평균
- **paired** 비교 (같은 input)이라 Wilcoxon signed-rank로 p-value
- 출력: model별 mean IoU ± std, p-value
- **null hypothesis**: 두 모델 IoU 분포 동일 → 거부하면 차이 의미 있음

### 축 2 — **Size-stratified IoU** (필수, 가설 핵심 검증)
- val sample을 **GT size**로 stratify: small (`w·h < 0.05`), medium, large
- 각 bucket에서 IoU 비교 + Wilcoxon
- **가설**: small bucket에서 Riemannian이 더 큰 격차로 우위
- 만약 차이가 size에 무관하다면 → 가설 약화 (다른 요인 의심)

### 축 3 — **K (ODE step) sensitivity** (선택)
- K = 2, 4, 8, 16, 32에서 IoU 측정
- **가설**: Riemannian이 작은 K에서도 **덜 무너짐** (chart space에서 직선)
- Euclidean은 K↓에서 IoU 급락, Riemannian은 완만 → 가설 지지

### 축 4 — **Init seed variance** (선택)
- 같은 image에 100개 다른 init seed로 ODE → IoU 분산 비교
- **가설**: Riemannian이 분산 작음 (chart geometry가 init noise에 덜 민감)

---

## 출력물 (검증 완료 시)

```
outputs/comparison/
├─ summary.json          # mean IoU, stratified IoU, K sweep 모두 모음
├─ stratified_iou.csv    # bucket × model 표
├─ k_sweep.csv           # K × model 표 (축 3만)
├─ seed_variance.csv     # seed × model × image 표 (축 4만)
├─ paired_test.txt       # statistical test 결과
├─ gif/
│  ├─ compare/sample_{i}.gif    # 가로 panel: Euclid | Riem (10 samples)
│  └─ per_index/{i}/...gif      # index별 trajectory
└─ figures/
   ├─ size_iou_curve.png     # size vs IoU (두 모델 overlay)
   ├─ k_sweep.png            # K vs IoU (축 3)
   └─ seed_variance.png      # box plot (축 4)
```

---

## 코드 모듈

```
inference/
├─ __init__.py
├─ metrics.py        # iou_xywh, paired_wilcoxon, stratify_by_size
├─ compare.py        # 두 ckpt 로드 → 같은 sample/seed → 축 1 + 2 표
├─ k_sweep.py        # 축 3 (선택)
├─ seed_var.py       # 축 4 (선택)
└─ visualize.py      # size_iou_curve, k_sweep, seed_variance 그래프
```

각 모듈 `__main__`에 dummy ckpt 1 sample 비교 → JSON 출력 형태 sanity.

---

## 결정 / 가정

### 확정
- ✅ plans/comparison.md 별도 파일 (visualization.md는 시각화 spec, 검증은 별 영역)
- ✅ ODE K = 10 (visualization.md §2 통일)
- ✅ 같은 init seed로 두 모델 같은 입력 받음 → paired 분석
- ✅ box-space IoU만 (chart 무관, 공정)
- ✅ 통계 검정: **Wilcoxon signed-rank** (IoU는 정규성 약함, paired)

### 확정 (추가)
- ✅ **풀세트 1+2+3+4** 진행 (사용자 결정)

### TBD
1. 검정 multiple-comparison 보정 (stratify 3 bucket 동시 검정 시 Bonferroni 등)?
2. 통계적 유의 + 효과 크기 임계값 (예: ΔIoU ≥ 0.01 absolute)?

---

## 작업 순서 (풀세트)
1. `inference/metrics.py` (TDD: IoU shape / 경계 케이스 / 알려진 값 + paired_wilcoxon)
2. `inference/compare.py` — 축 1 + 2 (TDD: dummy ckpt smoke + JSON)
3. **두 ckpt 비교 실행** → `outputs/comparison/summary.json` + `stratified_iou.csv`
4. `inference/k_sweep.py` — 축 3 (K=2,4,8,16,32) → `k_sweep.csv` + `figures/k_sweep.png`
5. `inference/seed_var.py` — 축 4 (100 seeds) → `seed_variance.csv` + `figures/seed_variance.png`
6. `inference/visualize.py` — `size_iou_curve`, `k_sweep`, `seed_variance` figure 생성
7. (별개) 비교 GIF (가로 panel: Euclid | Riem) + per-index GIF — `inference/gifs.py`

---

## 다음 단계 전환 조건 (Phase 2 → 마무리)
- 축 1+2 결과 확보
- Riemannian 우세이면: 가설 확인 (특히 small box bucket)
- Euclidean 우세 또는 무차이이면: 결과 그대로 보고 (toy 규모/데이터셋 한계 등 코멘트)
- 결과 정리 후 `plans/training.md`, `plans/comparison.md` → `plans/archive/`
