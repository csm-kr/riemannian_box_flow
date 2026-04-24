# riemannian_box_flow

## Docker 실행

### 최초 빌드 및 백그라운드 실행
```bash
docker compose up -d --build
```

### 이후 실행 (빌드 생략)
```bash
docker compose up -d
```

### 컨테이너 접속
```bash
docker exec -it bflow_dev bash
```

### 종료
```bash
docker compose down
```

### cv2.imshow() 사용 시 (X11 forwarding)
```bash
xhost +local:docker
docker compose up -d
```
