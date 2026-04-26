# TODO.md

## Now
- [ ] `model/charts/signal.py` 구현
      → `box_to_signal(b) = 3(2b-1)`, `signal_to_box(s) = (s/3 + 1)/2`
      → TDD: round-trip 테스트, edge case (0, 1) 검증

## Next (Phase 1 — Euclidean only)
- [ ] `model/components/time_embed.py` (sinusoidal + 2-layer MLP)
- [ ] `model/components/rope2d.py` (2D RoPE)
- [ ] `model/components/image_encoder.py` (DINOv2 ViT-S/14, frozen)
- [ ] `model/components/dit_block.py` (self-attn + cross-attn + adaLN)
- [ ] `model/backbone.py` (chart 무관 backbone 조립)
- [ ] `model/flow_signal.py` — `SignalFlowModel`
      → forward(s_t, t, image) → v_pred
      → loss: FM MSE, target = s_1 - s_0
      → `s_0 ~ clip(N(0,I), -3, 3)` 샘플링
      → endpoint decode: ŝ_1 = s_t + (1-t)û → b̂_1
- [ ] Phase 1 sanity check
      → 1-step training이 loss 감소 / forward/backward 통과
      → 작은 데이터로 overfit 시 GIF trajectory 합리성 확인

## Future (Phase 2 — Riemannian, Phase 1 검증 후)
- [ ] `model/charts/global_chart.py`
- [ ] `model/charts/local_chart.py`
- [ ] `model/flow_global.py` — `GlobalChartFlowModel`
- [ ] `model/flow_local.py` — `LocalChartFlowModel`
- [ ] Euclidean vs Riemannian trajectory 시각 비교

## Done
- [x] dataset/ 모든 모듈 구현
- [x] `python -m dataset.mnist_box_dataset` sanity check 통과
- [x] `plans/dataset.md` archive 이동
- [x] `plans/model.md` Phase 1 / Phase 2 분리 + Euclidean 학습 스펙 확정
