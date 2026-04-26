# plans/training.md — Training 설계 + 계획

## 목표 (Phase 1)
MNIST 10-box flow matching **Euclidean(signal)** 학습 + ODE inference.

- 학습: signal-space 직선 flow matching (model.md §1 사양)
- 추론: ODE Euler 적분 **K ∈ [10, 30]** step
- 결과: index 0~9 box trajectory `b_0 → b_1` GIF 생성

Riemannian (Phase 2)는 별도 학습 스크립트 없이 같은 trainer를 재활용 — chart 객체만 교체.

---

## 1. 입력 / 출력 스키마

### Dataset가 주는 것 (`MNISTBoxDataset.__getitem__`)
| key             | shape        | space          |
|-----------------|--------------|----------------|
| image           | (3, 224, 224)| RGB [0,1]      |
| gt_boxes        | (10, 4)      | box [0,1]^4    |
| init_boxes      | (10, 4)      | box [0,1]^4    |
| gt_signals      | (10, 4)      | signal [-3,3]^4|
| init_signals    | (10, 4)      | signal [-3,3]^4|
| labels          | (10,)        | 0..9           |

> Phase 1 학습은 `gt_boxes`(또는 `gt_signals`)와 `image`만 사용.
> `init_signals`는 dataset가 미리 sample 해 두었지만, 학습 step에서는 매번 fresh `s_0 ~ clip(N(0,I), -3, 3)`을 새로 뽑는다 (FM stochasticity 확보).

### Model이 받는/주는 것
- in: `s_t (B, 10, 4)`, `t (B,)`, `image (B, 3, 224, 224)`
- out: `v_pred (B, 10, 4)` (signal-space tangent)

---

## 2. 학습 step

```python
# batch
b_1   = batch["gt_boxes"]                      # (B, 10, 4)
image = batch["image"]                         # (B, 3, 224, 224)
B     = b_1.shape[0]

# Phase 1 — Signal flow
s_1 = box_to_signal(b_1)                       # (B, 10, 4)
eps = torch.randn_like(s_1)
s_0 = eps.clamp(-3, 3)                         # clipped Gaussian
t   = torch.rand(B, device=s_1.device)         # (B,)

t_b = t.view(B, 1, 1)
s_t = (1 - t_b) * s_0 + t_b * s_1

u_target = s_1 - s_0
u_pred   = model(s_t, t, image)                # SignalFlowModel

loss_fm = F.mse_loss(u_pred, u_target)

# (옵션) box-space aux
loss = loss_fm
if lambda_box > 0:
    s_hat_1 = s_t + (1 - t_b) * u_pred
    b_hat_1 = signal_to_box(s_hat_1)
    loss = loss + lambda_box * (l1(b_hat_1, b_1) + giou_loss(b_hat_1, b_1))
```

`λ_box`는 0으로 시작 (FM-only). 학습 안정 후 0.1~1.0 범위에서 실험.

---

## 3. Optimizer / schedule (시작값)

| 항목 | 값 |
|------|----|
| optimizer | AdamW |
| lr | 1e-4 |
| weight_decay | 1e-4 |
| betas | (0.9, 0.999) |
| schedule | linear warmup (1k step) → cosine decay |
| grad clip | 1.0 |
| batch size | 64 |
| 학습 step | 50k (소규모 sanity); 필요 시 확장 |
| AMP | bf16 (가능 시) |

DINOv2 인코더는 frozen이라 trainable params만 옵티마이저에 등록.

---

## 4. ODE inference (K-step Euler)

```python
@torch.no_grad()
def sample(model, image, K=16, device="cuda"):
    """image: (B, 3, 224, 224) → predicted boxes (B, 10, 4)"""
    B = image.shape[0]
    s = torch.randn(B, 10, 4, device=device).clamp_(-3, 3)
    dt = 1.0 / K
    traj_signal = [s.clone()]
    for k in range(K):
        t = torch.full((B,), k * dt, device=device)
        v = model(s, t, image)
        s = s + dt * v
        traj_signal.append(s.clone())
    boxes = signal_to_box(s).clamp_(0, 1)              # (B, 10, 4)
    traj_boxes = [signal_to_box(z).clamp_(0, 1) for z in traj_signal]
    return boxes, traj_boxes
```

### 4.1 K (step 수)
- 기본 **K = 16** (10~30 범위 중간)
- inference config로 노출 — 실험에서 K=10, 16, 30 비교

### 4.2 Solver
- 시작은 **Euler** (단순, 분석 용이)
- 추후 RK4 옵션 추가 가능 (TBD)

### 4.3 boundary clipping
- 마지막 box는 `[0,1]` 안에 들어와야 시각화가 깔끔 → `clamp_(0,1)` 적용
- 중간 trajectory frame도 시각화 시 동일 clamp

---

## 5. 검증 / metric

매 N step마다 val split에서:
- **L_FM** (val loss)
- **mean IoU** (predicted `b̂_1` vs `gt_boxes`, index-wise — class identity 고정이라 Hungarian 불필요)
- **mean GIoU**
- 샘플 이미지 1~2개 trajectory GIF 저장

---

## 6. 시각화 / GIF

```
frame_k: canvas + draw boxes (b_0, b_1, ..., b_K)
         color per index (10 색)
         optionally GT box를 dashed로 overlay
save as: outputs/gif/{step:06d}_{idx}.gif
```

각 sample마다 K+1 frame.

---

## 7. 체크포인트 / 로깅

- 체크포인트: `outputs/ckpt/step_{N}.pt` (model state_dict + optimizer + step)
- 마지막 best val loss는 별도 `best.pt`
- 로깅: TensorBoard (`outputs/tb/`), 이미 docker-compose에서 6006 포트 노출

---

## 8. 파일 구조 (예정)

```
training/
├─ __init__.py
├─ config.py            ← dataclass config (lr, K, λ_box, ...)
├─ losses.py            ← fm_loss, l1_box, giou_loss
├─ ode_sampler.py       ← Euler K-step (RK4는 future)
├─ metrics.py           ← iou, giou (val용)
├─ trainer.py           ← train loop, val, ckpt, tb
└─ train.py             ← entry point (config 로드 + run)
```

`__main__`에서 sanity check: 작은 batch로 1-step forward/backward, ODE 1-step 동작 확인.

---

## 9. 학습 절차 (단계별)

1. **Forward/backward sanity** — random batch 1개로 `loss.backward()` 통과
2. **Overfit micro-set** — 1 batch 100회 반복 → loss 거의 0, predicted box ≈ GT box 확인
3. **Full train (50k step)** — train loss / val loss 곡선 + GIF 출력
4. **K 비교** — K=10, 16, 30 inference 결과 비교
5. **(옵션) λ_box 도입** — FM loss plateau 후 box aux 추가 효과

---

## 10. 결정사항 (확정)
1. ✅ Phase 1 = Euclidean signal flow만 학습
2. ✅ s_0는 학습 step마다 fresh sampling (`clip(N(0,I), -3, 3)`)
3. ✅ inference는 ODE Euler, K ∈ [10, 30] (default 16)
4. ✅ optimizer = AdamW, lr 1e-4, warmup 1k + cosine
5. ✅ class identity 고정 (Hungarian 불필요)
6. ✅ DINOv2 frozen, trainable params만 optimize

---

## 11. TBD (구현 단계)
- 정확한 batch size / step 수 (GPU 메모리 보고 조정)
- bf16 vs fp32 (DINOv2 안정성 확인)
- λ_box 도입 시점 / 값
- RK4 solver 추가 여부
- val 빈도 / GIF 저장 빈도
- DINOv2 patch token 정확한 수 (model.md §6 TBD와 동일)

---

## 12. 다음 단계 전환 조건 (Visualization phase로)
- Phase 1 학습이 train/val loss 수렴 + 합리적 box 예측
- ODE K-step trajectory GIF 생성 가능
- `plans/training.md` → `plans/archive/`
