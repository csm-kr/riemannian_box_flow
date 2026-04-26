# TODO.md

## Now
- [ ] TensorBoard writer를 `training/trainer.py`에 통합
      → `outputs/{run_name}/tb/`에 train/val loss, lr 로깅
      → GIF는 `add_image` 또는 그대로 파일로
      → CLI `--run-name` 추가 (기본: `full_run`)
- [ ] **Docker container 재시작** (shm_size 8gb 적용 — `docker compose down && docker compose up -d`)
- [ ] TB 띄우기 (`tensorboard --logdir outputs --bind_all`, 6006 포트는 docker-compose에 이미 노출)
- [ ] **Full train 실행** (50k step, batch 64, hidden 256, depth 6, DINOv2 frozen)
      → ~1시간 예상 (RTX 6000, 14 step/s 기준)
      → 후반 GIF 합리성 확인 (predicted box 위치/크기가 GT와 가까워지는지)

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
