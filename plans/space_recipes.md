# plans/space_recipes.md — Space별 학습 recipe + init bound

각 실험의 **box ↔ space 변환 / 학습 1 step / init 분포·range**를 한 곳에 정리.
구현은 `model/trajectory.py` (encoder/decoder + `sample_init_box`) + `model/flow_*.py`.

---

## 0. 공통 init (`sample_init_box`)

모든 모델은 **box space에서 b_0을 샘플링**한 뒤 각자 chart로 인코딩.
→ space 비교가 init prior에 휘둘리지 않음.

### 0.1 prior `"default"` (Phase 1/2 기본)
```
ε ~ N(0, I_4)
s   = clip(ε, −3, 3)
b_0 = (s + 3) / 6                # 4성분 모두 동일 분포
```
- box-space bound: **b_0 ∈ [0, 1]** (clip 경계 → 0 또는 1)
- 실제 분포: 평균 0.5, std ≈ 1/6 ≈ 0.167 → 대부분 [0.33, 0.67]
- 꼬리에서 size 0 또는 1 가능 → `eps=1e-3` 클램프 (chart/logit) 필요

### 0.2 prior `"small_size"` (실험 010~012)
```
pos: default와 동일 → b_pos ∈ [0, 1]
w, h ~ U[0.01, 0.05]              # size만 작게 강제
```
- 위치는 균등에 가깝고 size는 강제로 small-box 분포 → small bucket IoU 분석용

---

## 1. Signal space (S-E, 실험 001)

### 1.1 변환
```
s = 6 b − 3                      # box [0,1]^4 → signal [−3, +3]^4
b = (s + 3) / 6
```
4성분(`cx, cy, w, h`) 모두 동일 affine. **bounded·symmetric.**

### 1.2 학습 1 step
```
b_1                                              # GT
b_0 = sample_init_box(default)                   # ∈ [0,1]
s_0 = 6 b_0 − 3                                  # ∈ [−3, +3]
s_1 = 6 b_1 − 3
t   ~ U(0, 1)
s_t = (1 − t) s_0 + t s_1
u   = s_1 − s_0                                  # constant in t
û  = v_θ(s_t, t, x)                              # model: signal → signal
L   = ‖ û − u ‖²
```

### 1.3 그 space에서의 init range (default prior)
| 성분 | 이론 bound | typical (std≈1) |
|---|---|---|
| s_0 (전 성분) | [−3, +3] | 평균 0, ±1 내 ~68% |

ODE inference: signal space에서 Euler, 마지막에 `b = (s+3)/6`.

---

## 2. Riemannian: psi-trajectory + signal-space model (S-R, 실험 002)

= "model은 signal space에서 작동, **trajectory만 chart psi에서 직선**"

### 2.1 chart psi
```
psi(b)     = (cx, cy, log max(w, eps), log max(h, eps))     # pos는 raw
psi_inv(y) = (y_cx, y_cy, exp(y_lw), exp(y_lh))
eps = 1e-3
```

### 2.2 학습 1 step
```
b_1, b_0 = ... (default init)
y_0 = psi(b_0),  y_1 = psi(b_1)                  # chart space
t   ~ U(0, 1)
y_t = (1 − t) y_0 + t y_1                        # chart 직선
b_t = psi_inv(y_t)
s_t = 6 b_t − 3                                  # 모델 입력은 signal

u_t = d s_t / dt   (분석식)
   ├─ u_cx = 6 (cx_1 − cx_0)                     # = Eucl과 동일
   ├─ u_cy = 6 (cy_1 − cy_0)
   ├─ u_w  = 6 w_t · log(w_1 / w_0)              # state-dep (b_t)
   └─ u_h  = 6 h_t · log(h_1 / h_0)

û = v_θ(s_t, t, x)                               # model: signal → signal
L  = ‖ û − u_t ‖²
```

### 2.3 그 space에서의 init range
- 모델 입력 `s_t` 범위는 **§1과 동일** ([−3, +3]) — `b_t = psi_inv(y_t)` 가 [0,1]에 머물기 때문.
- 단 chart space `y_0` 자체:
  - `y_pos ∈ [0, 1]`
  - `y_siz = log(b_siz) ∈ [log eps, 0] = [−6.9, 0]`, default 분포에서 typical [−1.1, −0.4]

ODE inference: signal space (모델과 동일).

---

## 3. Chart-native (C-R, 실험 003)

= "model도 chart space에서 작동, target도 chart velocity (constant)"

### 3.1 학습 1 step
```
b_1, b_0 = ... (default init)
y_0 = psi(b_0),  y_1 = psi(b_1)
t   ~ U(0, 1)
y_t = (1 − t) y_0 + t y_1
u   = y_1 − y_0                                  # chart velocity, constant
û  = v_θ(y_t, t, x)                              # model: chart → chart
L   = ‖ û − u ‖²
```

### 3.2 그 space에서의 init range (default prior)
| 성분 | 이론 bound | typical |
|---|---|---|
| y_pos (cx, cy) | [0, 1] | 평균 0.5, std≈0.167 |
| y_siz (log w/h) | [−6.9, 0] | 평균 ≈ −0.7, 꼬리 −∞ 방향 |

→ **pos vs size scale 불균형 (1 vs 7)**. 학습 신호 imbalance 가능.

ODE inference: **chart space**에서 Euler, 마지막 `b = psi_inv(y)`.

---

## 4. Logit-native (실험 013, 우승)

= "4성분 모두 logit (symmetric, unbounded), model도 logit space"

### 4.1 변환
```
y = logit(b) = log(b / (1 − b))                  # b ∈ [eps, 1−eps]
b = sigmoid(y) = 1 / (1 + exp(−y))
eps = 1e-3
```
**4성분(cx, cy, w, h) 모두 동일 변환.** symmetric & unbounded.

### 4.2 학습 1 step
```
b_1, b_0 = ... (default init)
y_0 = logit(b_0),  y_1 = logit(b_1)
t   ~ U(0, 1)
y_t = (1 − t) y_0 + t y_1
u   = y_1 − y_0                                  # logit velocity, constant
û  = v_θ(y_t, t, x)                              # model: logit → logit
L   = ‖ û − u ‖²
```

### 4.3 그 space에서의 init range (default prior)
| 성분 | 이론 bound | typical |
|---|---|---|
| y (전 성분) | [logit(1e-3), logit(0.999)] = [−6.9, +6.9] | 평균 0, 대부분 [−2, +2] |

- `logit(0.5) = 0` → default prior (평균 0.5) 중심.
- pos·size 모두 같은 분포 → **scale 균형** + signal과 비슷한 magnitude.

ODE inference: logit space에서 Euler, 마지막 `b = sigmoid(y)`.

---

## 5. 정리표

| Space | encode | decode | model 입력 | 학습 target | constant in t? | 4성분 대칭 | bounded |
|---|---|---|---|---|---|---|---|
| Signal (S-E) | `6b−3` | `(s+3)/6` | s | `s_1 − s_0` | ✅ | ✅ | ✅ ±3 |
| psi+S (S-R) | psi | psi_inv | s (b_t 거쳐) | `ds/dt` (state-dep) | ❌ | ❌ | ✅ |
| Chart (C-R) | psi | psi_inv | y | `y_1 − y_0` | ✅ | ❌ (pos vs log size) | ❌ size 한쪽만 |
| **Logit** | `logit b` | `sigmoid y` | y | `y_1 − y_0` | ✅ | ✅ | ❌ unbounded |

**우승 조합 (logit_native, 0.815)** = constant target + 4성분 대칭 + signal-급 magnitude.

---

## 6. small_size init prior (각 space에서의 range)

| Space | size 성분의 init range |
|---|---|
| box | [0.01, 0.05] |
| signal | `6·[0.01,0.05] − 3 = [−2.94, −2.7]` |
| chart psi `log w` | `[log 0.01, log 0.05] = [−4.6, −3.0]` |
| logit `logit w` | `[logit 0.01, logit 0.05] ≈ [−4.6, −2.94]` |

→ small_size prior는 size 성분만 **음의 영역으로 강제 이동** (logit/chart 모두). 위치는 default와 동일.
