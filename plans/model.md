# plans/model.md — Model 설계 + 계획

## 목표
10-box flow matching 모델 `v_θ(s_t, t, x)` 구현.
- box space `b = (cx, cy, w, h) ∈ [0,1]^4`
- 이미지 조건: 224×224 MNIST canvas
- `t ~ U(0,1)`
- **Phase 1: Euclidean (signal space) flow만 학습** ← 우선
- **Phase 2: Riemannian (global / local chart) — Phase 1 학습 가능성 확인 후**

---

## 0. Phase 계획

| Phase | 좌표계 | 모델 | 우선순위 |
|-------|--------|------|---------|
| 1     | Signal (Euclidean) | `SignalFlowModel` | **현재** |
| 2     | Global chart (Riemannian) | `GlobalChartFlowModel` | 보류 |
| 2     | Scale-aware local chart (Riemannian) | `LocalChartFlowModel` | 보류 |

> Phase 1에서 학습 가능성 / loss 수렴 / GIF trajectory 합리성을 먼저 확인하고,
> 그 다음에 같은 backbone으로 Riemannian variant를 붙여 비교한다.

---

## 1. Phase 1: Euclidean (Signal) 학습 스펙

### 1.1 표기
- Box space: `B = [0,1]^4`, `b = (c_x, c_y, w, h)`
- Signal space: `S = [-3, 3]^4`
- Box ↔ signal map:
  ```
  ϕ_S(b)    = 3 (2b - 1)              # box → signal
  ϕ_S^{-1}(s) = (s/3 + 1) / 2          # signal → box
  ```

### 1.2 학습 1 step (수식)

```
1) Data
   b_1 ∈ B            # GT box (B, 10, 4)

2) Box → signal target
   s_1 = ϕ_S(b_1) = 3 (2 b_1 - 1)

3) Init signal sampling (signal space에서 직접)
   ε ~ N(0, I_4)
   s_0 = clip(ε, -3, 3)

4) Time
   t ~ U(0, 1)

5) 중간 signal (선형 보간)
   s_t = (1 - t) s_0 + t s_1

6) Target velocity (signal-space 직선)
   u^S = s_1 - s_0
        = (s_1 - s_t) / (1 - t)        # 동치

   ※ signal flow는 직선이라 t에 무관.

7) Model prediction
   û = v_θ(s_t, t, x)                  # x = image feature / condition

8) Flow matching loss
   L_FM = E[ ‖ v_θ(s_t, t, x) - (s_1 - s_0) ‖_2^2 ]

9) Endpoint decoding
   ŝ_1 = s_t + (1 - t) û
   b̂_1 = ϕ_S^{-1}(ŝ_1) = (ŝ_1 / 3 + 1) / 2
```

### 1.3 (옵션) Box-space auxiliary loss

```
L_box = ‖ b̂_1 - b_1 ‖_1  +  L_GIoU(b̂_1, b_1)
L     = L_FM + λ_box · L_box
```

`λ_box`는 처음 0으로 두고 (FM-only) 학습 안정 후 조정.

### 1.4 한 줄 요약

```
b_1 ─ϕ_S─► s_1
s_0 ~ clip(N(0,I), -3, 3)
s_t = (1-t)s_0 + t s_1
v_θ(s_t, t, x) ≈ s_1 - s_0
```

---

## 2. 모델 구조 (chart 무관, Phase 1/2 공통)

```
        image (B, 3, 224, 224)
              │
       [DINOv2 ViT-S/14]   ← pretrained, frozen 시작 (LoRA TBD)
              │
     patch tokens (B, N_p, C_dino=384)
              │
         [Adapter MLP]      ← C_dino → C_model
              │
     ┌────────▼────────┐
     │   DiT blocks    │ ×L
s_t ─►│  - lift to     │
     │    latent dim   │
     │  - self-attn    │
     │    (10 queries) │
     │  - cross-attn   │ ◄── image patches (RoPE-2D)
     │  - adaLN(t)     │ ◄── t embedding
     └────────┬────────┘
              │
     [Final projection]
              │
         v_out (B, 10, 4)
```

### 2.1 구성요소
- **Image Encoder**: DINOv2 ViT-S/14, pretrained, frozen 시작
- **Box Query**: learnable embedding `(10, C_model)` — index 0~9 = class identity
- **DiT block**: self-attn(queries) + cross-attn(patches with 2D RoPE) + adaLN(t)
- **t embedding**: sinusoidal → 2-layer MLP
- **2D RoPE**: image patch 측 key/value에만 적용
- **Final proj**: `(10, C_model) → (10, 4)`

### 2.2 사이즈 (시작 기준)

| 항목 | 값 |
|------|-----|
| image encoder | DINOv2 ViT-S/14 (pretrained, frozen) |
| C_dino | 384 |
| C_model | 256 |
| DiT depth | 6 |
| heads | 8 |
| n_queries | 10 |

---

## 3. 학습 루프 의사코드 (Phase 1)

```python
# Inputs: b_1 (B, 10, 4), image (B, 3, 224, 224)
B = b_1.shape[0]

s_1 = 3 * (2 * b_1 - 1)                          # (B, 10, 4)
eps = torch.randn_like(s_1)
s_0 = eps.clamp(-3, 3)
t   = torch.rand(B, device=s_1.device)           # (B,)

t_b = t.view(B, 1, 1)
s_t = (1 - t_b) * s_0 + t_b * s_1                # (B, 10, 4)

u_target = s_1 - s_0                             # (B, 10, 4)
u_pred   = model(s_t, t, image)                  # (B, 10, 4)

loss_fm = F.mse_loss(u_pred, u_target)

# (옵션) box-space aux
# s_hat_1 = s_t + (1 - t_b) * u_pred
# b_hat_1 = (s_hat_1 / 3 + 1) / 2
# loss_box = l1(b_hat_1, b_1) + giou_loss(b_hat_1, b_1)
# loss = loss_fm + lambda_box * loss_box
loss = loss_fm
```

추론 / 시각화는 `s_0 ~ clip(N(0,I))` 에서 시작해 ODE Euler integration → `ϕ_S^{-1}`.

---

## 4. 파일 구조 (Phase 1만 우선 구현)

```
model/
├─ __init__.py
├─ components/
│  ├─ __init__.py
│  ├─ image_encoder.py     ← DINOv2 wrapper                       [P1]
│  ├─ rope2d.py            ← 2D RoPE                              [P1]
│  ├─ dit_block.py         ← self-attn + cross-attn + adaLN       [P1]
│  └─ time_embed.py        ← t embedding                          [P1]
├─ charts/
│  ├─ __init__.py
│  ├─ signal.py            ← ϕ_S, ϕ_S^{-1}                        [P1]
│  ├─ global_chart.py      ← Riemannian global                    [P2]
│  └─ local_chart.py       ← Riemannian scale-aware local         [P2]
├─ backbone.py             ← chart 무관 backbone (DINOv2 + DiT)    [P1]
├─ flow_signal.py          ← SignalFlowModel + train/sampling     [P1]
├─ flow_global.py          ← GlobalChartFlowModel                 [P2]
└─ flow_local.py           ← LocalChartFlowModel                  [P2]
```

`[P1]`만 Phase 1에서 구현. `[P2]`는 Phase 1 학습 검증 후 착수.

---

## 5. 결정사항 (확정)
1. ✅ Phase 1은 Euclidean(signal)만 학습 — Riemannian은 Phase 2
2. ✅ DINOv2 pretrained weight 로드 (frozen 시작)
3. ✅ box query는 learnable embedding (10개 = class identity)
4. ✅ chart 별도 클래스 (Signal / Global / Local 분리, backbone 공유)
5. ✅ `s_0 ~ clip(N(0, I), -3, 3)` (signal space에서 직접 샘플)
6. ✅ Phase 1 loss = `L_FM` (MSE), `L_box`는 보조 옵션

---

## 6. TBD (Phase 1 구현 시)
- DINOv2 patch token 정확한 수 (register/CLS, 224 입력 grid)
- LoRA 적용 여부 (frozen으로 시작)
- 모델 입력 표현: `(s_t, t, image)`만? 또는 `b_t = ϕ_S^{-1}(s_t)`도 같이 줄지
- ODE solver (Euler vs RK4), 추론 step 수
- `λ_box` 도입 시점 / 값

---

## 7. Riemannian (Phase 2 reference) — 보류

> Phase 1 검증 후 활성화. 아래는 설계 문서로만 보존.

### 7.1 좌표계 비교표

| 항목         | Signal (Euclidean) [P1]    | Global chart (Riemannian) [P2] | Scale-aware local chart (Riemannian) [P2] |
|--------------|----------------------------|--------------------------------|-------------------------------------------|
| point        | `s = 3(2b - 1)`            | `r = φ(b)`                     | `ψ_{b_t}(b)`                              |
| size 처리    | `3(2w - 1)`                | `log w`                        | `log(w / w_t)`                            |
| center 처리  | `3(2c_x - 1)`              | `3(2c_x - 1)`                  | `(c_x - c_x^t) / w_t`                     |
| 기준점       | 없음 (global)              | global origin                  | 현재 box `b_t`                            |
| velocity     | `s_1 - s_0`                | `(r_1 - r_t) / (1 - t)`        | `ψ_{b_t}(b_1) / (1 - t)`                  |
| 의미         | 단순 숫자 이동             | log-size chart 이동            | 현재 box 크기 기준 상대 이동              |

> Local chart velocity는 원래 `[ψ_{b_t}(b_1) - ψ_{b_t}(b_t)] / (1-t)`이지만, `ψ_{b_t}(b_t) = 0`이므로 표에는 단순화 식.

### 7.2 Global chart
```
forward:   r = (3(2c_x-1), 3(2c_y-1), log w, log h)
inverse:   c_x = (r_cx/3 + 1)/2 ; w = exp(r_w)
target u:  u = (r_1 - r_t)/(1-t) = r_1 - r_0
decode:    r_pred = r_t + (1-t)*û  →  b̂_1 = inverse(r_pred)
```

### 7.3 Scale-aware local chart
```
ψ_{b_t}(b) = ((c_x - c_x^t)/w_t, (c_y - c_y^t)/h_t, log(w/w_t), log(h/h_t))

target u:
  u_x = (c_x^1 - c_x^t) / ((1-t) w_t)
  u_y = (c_y^1 - c_y^t) / ((1-t) h_t)
  u_w = log(w_1/w_t) / (1-t)
  u_h = log(h_1/h_t) / (1-t)

decode (Exp_{b_t}):
  ĉ_x^1 = c_x^t + (1-t) w_t û_x
  ĉ_y^1 = c_y^t + (1-t) h_t û_y
  ŵ_1   = w_t exp((1-t) û_w)
  ĥ_1   = h_t exp((1-t) û_h)
```
