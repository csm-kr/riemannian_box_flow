# plans/visualization.md — Visualization 설계 + 계획

## 목표
1. **Per-index trajectory GIF**: index 0~9 box 각각의 `b_0 → b_1` ODE trajectory를 별도 GIF로 저장.
2. **(옵션) 통합 GIF**: 한 image 위에 10 box trajectory를 동시에 그린 GIF.
3. **Euclidean vs Riemannian 비교 GIF**: 같은 image / 같은 init seed에서 두 baseline의 trajectory를 가로 panel(224×448)로 비교.

---

## 1. 입력 / 출력
| 입력 | shape / 의미 |
|------|--------------|
| `image` | `(3, 224, 224)` |
| `traj_boxes` | `K+1` 개, 각 `(10, 4)` in `[0,1]` (모든 index 함께) |
| `gt_boxes` | `(10, 4)` (옵션, dashed overlay) |

→ 출력은 BGR uint8 frame list 또는 GIF 파일.

---

## 2. ODE 시각화 timestep
- 사용자 spec: `t = 0.0, 0.1, ..., 1.0` → **K = 10** (11 frames 포함 시작 포함)
- 학습 / 비교 inference 기본도 K=10로 통일.

---

## 3. Per-index GIF
같은 trajectory에서 index `i`의 box만 추출해 별도 GIF.

```
for i in range(10):
    frames_i = draw_trajectory_frames(image, [b[i] for b in traj_boxes], gt_boxes=gt[i])
    save_gif(frames_i, out_dir / f"per_index/{i:02d}.gif", fps=6)
```

색상은 `BOX_COLORS[i]` 그대로 (현재 `training/visualize.py` 정의 재활용).

---

## 4. 통합 GIF (옵션)
현재 `training/visualize.py:draw_trajectory_frames`가 이미 10 box 모두 한 frame에 그림. 그대로 사용.

---

## 5. Euclidean vs Riemannian 비교 GIF (Phase 2 핵심)

같은 image, **같은 `s_0` seed**에서 두 모델로 ODE inference → 가로로 합친 frame 시퀀스.

```
+-----------------+-----------------+
|   Euclidean     |   Riemannian    |   ← 224×448 BGR uint8
| (10 boxes,      | (10 boxes,      |
|  GT dashed)     |  GT dashed)     |
| t=0.0, 0.1, ... | t=0.0, 0.1, ... |
+-----------------+-----------------+
```

구현:
```python
frames_e = draw_trajectory_frames(image, traj_e, gt_boxes=gt)   # Euclidean
frames_r = draw_trajectory_frames(image, traj_r, gt_boxes=gt)   # Riemannian
frames   = [np.concatenate([fe, fr], axis=1) for fe, fr in zip(frames_e, frames_r)]
# label "Euclidean" / "Riemannian"를 각 panel 좌상단에 cv2.putText
save_gif(frames, out_dir / "compare/sample_{i}.gif", fps=6)
```

같은 init seed 보장:
```python
torch.manual_seed(seed)
s0 = torch.randn(B, 10, 4).clamp_(-3, 3)
# 두 모델 모두 이 s0으로 ODE 시작
```

---

## 6. 파일 구조
```
training/
├─ visualize.py    ← 현재 frame/GIF 빌더; per-index 헬퍼 추가
└─ ...

inference/         ← 새 모듈 (Phase 2 비교용)
└─ compare.py      ← 두 ckpt 로드 → 같은 sample / seed로 ODE → 가로 panel GIF
```

> `inference/`는 새로 추가. Phase 2 비교 inference 전용.
> 단순한 비교 한 번이면 `training/visualize.py`에 함수만 추가해도 됨 — 분리 여부는 구현 시점에 결정.

---

## 7. `__main__` sanity check (모듈 별)
- `training/visualize.py`: 이미 random sample → GIF 저장 통과 (`sanity_traj.gif`)
- `inference/compare.py` (생길 경우): dummy 모델 2개 + 같은 seed → frame shape `(224, 448, 3)`, GIF 저장 / 파일 크기 > 0

---

## 8. 결정 / TBD
1. ✅ K = 10 (사용자 spec)
2. ✅ 가로 panel 비교 (Euclidean | Riemannian)
3. ✅ Per-index GIF는 별도 폴더 `per_index/{i:02d}.gif`
4. TBD — overlay 방식 비교 GIF도 같이 만들지 (추가 비용 거의 없음)
5. TBD — `compare.py` 별도 모듈 여부 (visualize.py 확장으로 충분할 수도)
