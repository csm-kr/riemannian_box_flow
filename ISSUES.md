# ISSUES.md

## 해결된 이슈
- [x] `nvidia/cuda:13.0` 태그 없음 → `13.0.3`으로 수정
- [x] Ubuntu 22.04 기본 nodejs v12로 Claude Code 설치 실패 → NodeSource Node.js 20 LTS로 교체
- [x] **DataLoader `unable to allocate shared memory`** — Docker 기본 `/dev/shm`이 64MB라 num_workers > 0일 때 collate가 shm 고갈
  - 임시: `torch.multiprocessing.set_sharing_strategy("file_system")` (`training/trainer.py` 상단)
  - 영구: `docker-compose.yml`에 `shm_size: "8gb"` 추가 (다음 build 시 적용)
- [x] **Full train (50k) shm 누수로 step 37000 (74%)에서 죽음**
  - 증상: `unable to allocate shared memory(shm) for file </torch_43012_..._0>` (`_dump_gif`의 `next(iter(val_loader))` 라인)
  - 원인: `_validate` / `_dump_gif`가 매 호출마다 `iter(val_loader)`로 새 워커를 spawn → `file_system` sharing 전략 하에서 `/tmp/torch_*` 누수가 누적되어 8GB shm 고갈 (74회 val + 37회 GIF × 2 worker)
  - 해결: `val_loader`는 `num_workers=0`, `train_loader`는 `persistent_workers=True`로 워커 재spawn 자체를 제거 (`training/trainer.py:_make_loaders`)
  - 결과: step_035000.pt를 final.pt로 사용 (val_loss 최저 0.0835 @ step 35500). 재학습은 다음 PR에서 검증.

## 열린 이슈
없음
