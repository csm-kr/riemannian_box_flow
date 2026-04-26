# TODO.md

## Now
- [ ] full train run (50k step, batch 64) + train/val 곡선 + 최종 GIF 확인

## Next
- [ ] K=10/16/30 비교 inference + GIF 비교
- [ ] (선택) bf16 AMP 안정성 확인
- [ ] Phase 2 (Riemannian) 착수 검토

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
- [x] `model/charts/signal.py` (TDD: 실패→구현→통과)
- [x] `plans/training.md` 작성 (Phase 1 학습/추론 계획)
- [x] `model/components/time_embed.py` (TDD: 실패→구현→통과)
- [x] `model/components/rope2d.py` (TDD: 실패→구현→통과)
- [x] `model/components/image_encoder.py` (DINOv2 ViT-S/14 wrapper, frozen)
- [x] `model/components/dit_block.py` (self-attn + cross-attn(RoPE) + adaLN)
- [x] `model/backbone.py` (DiT 조립)
- [x] `model/flow_signal.py` (Phase 1 `SignalFlowModel` + ODE sample)
- [x] `training/visualize.py` (trajectory → frames → GIF)
- [x] `training/{config,trainer,train}.py` (FM-only Phase 1 trainer)
- [x] Phase 1 smoke (50 step) + short run (1000 step) — loss 4.53→1.23, val 1.47→1.17
