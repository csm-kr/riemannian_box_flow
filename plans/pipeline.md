# plans/pipeline.md — 4 Baseline 데이터 파이프라인 비교

각 baseline의 **한 학습 step에서 만들어지는 모든 텐서** (`b_*`, `s_*`, `c_*`, `u_*`, `loss`) 를 shape와 함께 정리.

## 표기

| 기호 | 의미 | 공간 | 범위 |
|---|---|---|---|
| `b` | box (cx, cy, w, h) | box | `[0, 1]^4` |
| `s` | signal | signal | `[-3, 3]^4`,  `s = 6b - 3` |
| `c` | chart | chart | `c = (cx, cy, log w, log h)` ε-clamp |
| `u_target` | 학습 target velocity | (학습이 생기는 공간) | constant 또는 state-dep |
| `u_pred` | 모델 prediction | (모델 출력 공간) | |
| `loss` | scalar | — | MSE(`u_pred`, `u_target`) |

`B` = batch size, `Q` = 10 (query / box 수)

## 공통 (Dataset → batch)

```
batch  =  next(DataLoader)

  image      ∈  (B, 3, 224, 224)        ← MNIST canvas
  b_1        ∈  (B, Q, 4)               ← gt_boxes  in [0, 1]^4
  ─ 학습은 image, b_1만 사용 ─
```

## 공통 좌표 변환 함수

```
signal_encode(b) = 6b - 3
signal_decode(s) = (s + 3) / 6

chart_encode(b)  = ( b[..., 0], b[..., 1],  log max(b[..., 2], ε),  log max(b[..., 3], ε) )
chart_decode(c)  = ( c[..., 0], c[..., 1],  exp(c[..., 2]),  exp(c[..., 3]) )

ε = 1e-3
```

## 공통 init `b_0` (모든 baseline)

```
s_0_seed   ∈  (B, Q, 4)        ← clip(N(0,I), -3, 3)              [signal]
b_0        =  signal_decode(s_0_seed)        ∈  (B, Q, 4)         [box, 0~1]
```

→ 4 baseline 모두 **이 b_0 분포에서 시작**. 자기 좌표계 변환만 다름.

`t  ~  U(0, 1)   ∈  (B,)` ← 모두 동일하게 sample.

---

## (1, 2) — Signal-state 모델 (S-E, S-R)

**공통 골격**: 모델 입출력 모두 signal space.

```
                image (B, 3, 224, 224)
                     │
          s_t  ─►  ┌───────────────┐
       (B,Q,4)     │  Backbone     │  ─►  u_pred ∈ (B,Q,4)  [signal]
                   │   (signal)    │
           t  ─►   └───────────────┘
       (B,)
```

### ▶ (1) S-E — signal trajectory, **constant u**

```
b_1                 ∈ (B,Q,4)   [box]    ← from batch
b_0                 ∈ (B,Q,4)   [box]    ← signal_decode(s_0_seed)

s_1 = signal_encode(b_1) = 6 b_1 - 3        ∈ (B,Q,4)  [signal]
s_0 = signal_encode(b_0) = 6 b_0 - 3        ∈ (B,Q,4)  [signal]      (= s_0_seed)

# linear interp in SIGNAL space
s_t      = (1 - t)·s_0  +  t·s_1            ∈ (B,Q,4)  [signal]      ──► model input

# target velocity
u_target = s_1 - s_0                        ∈ (B,Q,4)  [signal]      ★ constant in t

# model
u_pred   = backbone(s_t, t, image)          ∈ (B,Q,4)  [signal]

# loss
loss     = mean( (u_pred - u_target)² )     ∈ scalar
```

### ▶ (2) S-R — chart-trajectory → signal, **state-dependent u**

```
b_1                 ∈ (B,Q,4)   [box]
b_0                 ∈ (B,Q,4)   [box]

c_1 = chart_encode(b_1) = (cx, cy, log w_1, log h_1)   ∈ (B,Q,4)  [chart]
c_0 = chart_encode(b_0)                                ∈ (B,Q,4)  [chart]

# linear interp in CHART space
c_t      = (1 - t)·c_0  +  t·c_1            ∈ (B,Q,4)  [chart]

# decode chart-state to box, then to signal (model input space)
b_t      = chart_decode(c_t)                ∈ (B,Q,4)  [box]
s_t      = signal_encode(b_t) = 6 b_t - 3   ∈ (B,Q,4)  [signal]      ──► model input

# target velocity = ds_t/dt analytically (model output space)
u_pos    = 6 · (b_1[..., :2] - b_0[..., :2])           ∈ (B,Q,2)
u_siz    = 6 · b_t[..., 2:] · (c_1[..., 2:] - c_0[..., 2:])
                                                       ∈ (B,Q,2)     ★ state-dep
u_target = cat(u_pos, u_siz)                ∈ (B,Q,4)  [signal]

# model
u_pred   = backbone(s_t, t, image)          ∈ (B,Q,4)  [signal]

# loss
loss     = mean( (u_pred - u_target)² )     ∈ scalar
```

### (1) vs (2) 핵심 차이

| | (1) S-E | (2) S-R |
|---|---|---|
| 직선 그어지는 공간 | signal | chart |
| 보조 텐서 | `s_0, s_1` | `c_0, c_1, c_t, b_t` 추가 |
| `u_target` 단위 | signal velocity | signal velocity |
| `u_target` 형태 | constant `s_1 - s_0` | `pos: 6·Δc`, `siz: 6·b_t·Δlog` (b_t 의존) |

---

## (3, 4) — Chart-state 모델 (C-R, C-E)

**공통 골격**: 모델 입출력 모두 chart space.

```
                image (B, 3, 224, 224)
                     │
          c_t  ─►  ┌───────────────┐
       (B,Q,4)     │  Backbone     │  ─►  u_pred ∈ (B,Q,4)  [chart]
                   │   (chart)     │
           t  ─►   └───────────────┘
       (B,)
```

### ▶ (3) C-R — chart trajectory, **constant u**

```
b_1                 ∈ (B,Q,4)   [box]
b_0                 ∈ (B,Q,4)   [box]

c_1 = chart_encode(b_1)                     ∈ (B,Q,4)  [chart]
c_0 = chart_encode(b_0)                     ∈ (B,Q,4)  [chart]

# linear interp in CHART space (= 모델 공간 그대로!)
c_t      = (1 - t)·c_0  +  t·c_1            ∈ (B,Q,4)  [chart]       ──► model input
                                                                         (그대로!)
# target velocity (모델 출력 공간 = chart)
u_target = c_1 - c_0                        ∈ (B,Q,4)  [chart]       ★ constant in t

# model
u_pred   = backbone(c_t, t, image)          ∈ (B,Q,4)  [chart]

# loss
loss     = mean( (u_pred - u_target)² )     ∈ scalar
```

### ▶ (4) C-E — box (cxcywh) trajectory → chart, **state-dependent u**

```
b_1                 ∈ (B,Q,4)   [box]
b_0                 ∈ (B,Q,4)   [box]

# linear interp in BOX (cxcywh) space  ◄── 모델 공간과 다름
b_t      = (1 - t)·b_0  +  t·b_1            ∈ (B,Q,4)  [box]

# encode current point to chart for model input
c_t      = chart_encode(b_t)                ∈ (B,Q,4)  [chart]       ──► model input

# target velocity = dc_t/dt analytically (chart output space)
u_pos    = b_1[..., :2] - b_0[..., :2]      ∈ (B,Q,2)                constant
u_siz    = (b_1[..., 2:] - b_0[..., 2:]) / b_t[..., 2:].clamp(min=ε)
                                            ∈ (B,Q,2)                ★ state-dep
                                                                       small w_t에서 발산!
u_target = cat(u_pos, u_siz)                ∈ (B,Q,4)  [chart]

# model
u_pred   = backbone(c_t, t, image)          ∈ (B,Q,4)  [chart]

# loss
loss     = mean( (u_pred - u_target)² )     ∈ scalar
```

### (3) vs (4) 핵심 차이

| | (3) C-R | (4) C-E |
|---|---|---|
| 직선 그어지는 공간 | chart (= 모델 공간) | box (cxcywh) |
| 보조 텐서 | `c_0, c_1` | `b_t` 추가 (box 보간) |
| `u_target` 형태 | constant `c_1 - c_0` | `pos: Δb_pos`, `siz: Δb_siz / b_t` (분모 발산) |
| 학습 안정성 | 높음 | **낮음** (3회 spike 5~18×) |

---

## (1,2) vs (3,4) — 본질 차이

| 측면 | (1,2) Signal-state | (3,4) Chart-state |
|---|---|---|
| `u_target` / `u_pred`가 사는 공간 | **signal** | **chart** |
| 매개 변수 (전형적) | `b_1, s_1, b_0, s_0, s_t, u, loss` | `b_1, c_1, b_0, c_0, c_t, u, loss` |
| 핵심 변환 | `signal_encode/decode` (affine) | `chart_encode/decode` (size에 log/exp) |
| `loss`가 직접 누르는 양 | signal-space velocity error | chart-space velocity error |
| inference 시 box 복원 | `signal_decode` (affine, 오차 선형) | `chart_decode` (exp, 오차 multiplicative) |
| box-IoU 평가 친화도 | **유리** | 불리 (exp 증폭) |

→ 같은 `b_0`에서 시작해도 **만들어지는 텐서 종류 자체가 (1,2)와 (3,4)는 다름**. (1,2)는 signal 공간 텐서를, (3,4)는 chart 공간 텐서를 산출해 학습.

---

## Inference 파이프라인 (참고)

학습 파이프라인은 위와 같지만, inference는 모두 ODE Euler K-step.

| baseline | 적분 공간 | step | 마지막 decode |
|---|---|---|---|
| (1) S-E, (2) S-R | **signal** | `s ← s + dt · u_pred` | `signal_decode` → `clamp(0,1)` |
| (3) C-R, (4) C-E | **chart**  | `c ← c + dt · u_pred` | `chart_decode`  → `clamp(0,1)` |

같은 b_0 (paired) 공유 시 → fair comparison.
