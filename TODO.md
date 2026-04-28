# TODO.md

## Now
없음.

## Next (Phase 3.0 후속 — 선택)
- [ ] (Opt-A) 013 wide-dataset 학습 후 boundary_audit / size_dynamics 재실행 — tiny / huge bucket 검증
- [ ] (Future) COCO 등 실제 detection 도메인에서 H2 (size dynamics) 격차 재현

## Next (마무리)
- [ ] plans archive 정리 (`plans/training.md`, `plans/comparison.md`, `plans/setup_analysis.md`, `plans/riem_strength.md`, `plans/logit_strengths.md` → archive)
- [ ] git PR 정리 (Phase 1/2/2.5/2.6/3.0 통합 또는 분리)
- [ ] (선택) hybrid 모델 4-panel 비교 GIF 생성
- [ ] (선택) 013 vs 007 hybrid paired 비교
- [ ] (선택) 013을 새 default baseline으로 README / PROJECT.md 업데이트

## Done (Phase 3.0 — H2)
- [x] **`inference/size_dynamics.py`** — 5-fine size bucket × 5 ratio bucket × {iou, size_err, center_err}
- [x] **`inference/size_dynamics_figures.py`** — 4 figures (iou_by_fine_size, size_err_by_fine_size, iou_by_ratio, iou_diff_heatmap)
- [x] **5000 sample paired run** — `outputs/logit_strengths/size_dynamics/`
- [x] **H2a 지지**: small bucket +0.0169 가장 큼, large +0.0040 으로 단조 감소
- [x] **H2b 지지**: ratio 4–8 +0.0155 (max), ≤1.5 +0.0046
- [x] **H2c 부분 지지** + **확장 발견**: size_err 미세 개선, **center_err 가 주된 개선** (small bucket −13%) — logit 의 `1/(x(1−x))` 는 4성분 모두 boundary-amplifying
- [x] **REPORT.md 종합 결론** — H1/H2 핵심 + reframed strength + practical implications

## Done (Phase 3.0 — H1)
- [x] **`inference/boundary_audit.py`** — raw ODE Euler + 4 metrics (coord_oob, canvas_oob, excess_l1, iou_clamp/no_clamp)
- [x] **K sweep on 5000 sample paired** — `outputs/logit_strengths/boundary_audit/`
- [x] **3 figures + REPORT.md H1 섹션** — K=2 에서 logit +0.048 IoU 격차가 핵심, canvas_oob 모든 K 에서 logit 낮음, coord_oob/excess 는 양쪽 0 (정직하게 reframing)

## Next (마무리)
- [ ] plans archive 정리 (`plans/training.md`, `plans/comparison.md`, `plans/setup_analysis.md`, `plans/riem_strength.md`, `plans/logit_strengths.md` → archive)
- [ ] git PR 정리 (Phase 1/2/2.5/2.6/3.0 통합 또는 분리)
- [ ] (선택) hybrid 모델 4-panel 비교 GIF 생성
- [ ] (선택) 013 vs 007 hybrid paired 비교
- [ ] (선택) 013을 새 default baseline으로 README / PROJECT.md 업데이트

## Future
- [ ] hybrid 모델을 size 변동이 큰 실데이터 (COCO 등)로 검증
- [ ] hybrid의 scale-relative metric 우위가 다른 도메인에서도 유지되는지

## Future
- [ ] (선택) Scale-aware Local chart 별도 실험 (`model.md §7.3`)
- [ ] (선택) bf16 AMP 안정성 확인
- [ ] (선택) DINOv2 patch token 정확한 수 확인 (`model.md §6 TBD`)

## Done
- [x] **exp 014 corner_logit_default** (`outputs/014_corner_logit_default/`, `outputs/comparison_corner_logit/REPORT.md`): 014 vs 001 mean IoU 0.7487 vs 0.8075 (−0.059, p≈0). 모든 size bucket에서 001 우위. corner-logit이 pos·size를 coupled로 만들어 학습 어려움 — "4성분 symmetric + 독립"이 logit chart 우승 조건이라는 가설 재확인.
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
