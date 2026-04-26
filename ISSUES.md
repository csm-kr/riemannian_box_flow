# ISSUES.md

## 해결된 이슈
- [x] `nvidia/cuda:13.0` 태그 없음 → `13.0.3`으로 수정
- [x] Ubuntu 22.04 기본 nodejs v12로 Claude Code 설치 실패 → NodeSource Node.js 20 LTS로 교체
- [x] **DataLoader `unable to allocate shared memory`** — Docker 기본 `/dev/shm`이 64MB라 num_workers > 0일 때 collate가 shm 고갈
  - 임시: `torch.multiprocessing.set_sharing_strategy("file_system")` (`training/trainer.py` 상단)
  - 영구: `docker-compose.yml`에 `shm_size: "8gb"` 추가 (다음 build 시 적용)

## 열린 이슈
없음
