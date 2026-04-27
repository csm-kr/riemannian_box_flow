# TODO.md

## Now
없음.

## Next
- [ ] (선택) 013 vs 007 hybrid paired 비교 (이전 2위였던 hybrid보다 얼마나 격차 있는지)
- [ ] (선택) 013 wide-dataset 학습 (`--wide-dataset`) — wide GT 분포에서도 logit chart 우위 확인 (005/006이 처참했던 영역)
- [ ] (선택) 013을 새 default baseline으로 README / PROJECT.md 업데이트

## Next
- [ ] plans archive 정리 (`plans/training.md`, `plans/comparison.md`, `plans/setup_analysis.md`, `plans/riem_strength.md` → archive)
- [ ] git PR 정리 (Phase 1/2/2.5/2.6 통합 또는 분리)
- [ ] (선택) hybrid 모델 4-panel 비교 GIF 생성

## Future
- [ ] hybrid 모델을 size 변동이 큰 실데이터 (COCO 등)로 검증
- [ ] hybrid의 scale-relative metric 우위가 다른 도메인에서도 유지되는지

## Future
- [ ] (선택) Scale-aware Local chart 별도 실험 (`model.md §7.3`)
- [ ] (선택) bf16 AMP 안정성 확인
- [ ] (선택) DINOv2 patch token 정확한 수 확인 (`model.md §6 TBD`)

## Done
- [x] **Phase 2 검증 5축 완료** (`outputs/comparison/REPORT.md`):
  - `inference/{metrics,compare,k_sweep,seed_var,visualize,gifs}.py` 작성
  - 5000 paired sample, 5 K-values, 100 seeds × 20 images, 10 비교 GIF
  - 결론: Eucl이 본 toy에서 일관 우위; Riem 약점은 size error (state-dependent velocity가 small box에서 학습 신호 약함)
- [x] **Riemannian 35k step 학습** — `outputs/002_riemannian/` (val 0.0951 최저, ckpt final.pt 확보)
- [x] **`model/flow_chart.py`** — `ChartFlowModel(SignalFlowModel)`, `fm_loss`만 riemannian_trajectory 사용 + trainer/CLI에 `--model {signal,chart}` 통합 (양쪽 8-step smoke 통과)
- [x] **`model/flow_signal.py` refactor** — `fm_loss`가 `euclidean_trajectory` 호출 (OLD vs NEW 5e-7 이내, 1-ULP 수준)
- [x] **`model/trajectory.py`** — signal_encode/decode + chart_encode/decode (psi) + euclidean/riemannian trajectory (TDD: 8 sanity 통과)
- [x] **Phase 1 마무리**: outputs 자동 numbering + TB 통합 + Docker shm 8gb + Full train 35k (val 0.0835 최저, final.pt 확보), shm 누수 패치 적용
- [x] **plans Phase 2 정리**: active.md / model.md §7.4 / visualization.md (사용자 spec 반영, chart `psi(b)=(cx,cy,log w,log h)` 단일)
- [x] dataset/ 모든 모듈 구현 + sanity 통과
- [x] `model/charts/signal.py` + `model/components/{time_embed,rope2d,image_encoder,dit_block}.py` + `model/backbone.py` + `model/flow_signal.py`
- [x] `training/{config,trainer,train,visualize}.py` (Phase 1 trainer, FM-only)
- [x] `plans/dataset.md` archive
