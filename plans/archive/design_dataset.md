# plan.md — Dataset 구현 계획

## 목표
`dataset.md` 스펙에 맞춰 **MNIST-10 Box Flow Toy Dataset**을 구현한다.
- 224x224 검은 canvas에 MNIST 숫자 0~9를 각 1개씩 배치
- `gt_boxes`, `init_boxes`, `gt_signals`, `init_signals` 생성
- 각 모듈은 `__main__`에서 sanity check 및 OpenCV 시각화 수행

---

## 폴더 구조
```
dataset/
├── __init__.py
├── mnist_source.py         # MNIST digit loader (torchvision)
├── box_utils.py            # box <-> signal 변환, overlap/bound 체크
├── sampler.py              # gt box / init signal 샘플러
├── canvas.py               # digit을 canvas에 paste
├── mnist_box_dataset.py    # torch Dataset 클래스 (메인)
└── visualize.py            # cv2 기반 sample 시각화
```

---

## 모듈별 역할 및 `__main__` 테스트

### 1. `mnist_source.py`
- `torchvision.datasets.MNIST`를 래핑해 class별 digit 이미지 반환
- `get_digit(label: int) -> np.ndarray (H, W)` API 제공
- `__main__`: 숫자 0~9 하나씩 로드해 shape / dtype 출력

### 2. `box_utils.py`
순수 함수 모음. 외부 의존성 최소화.
- `signal_to_box(s)`: `[-3, 3]^4 -> [0,1]^4`
- `box_to_signal(b)`: `[0,1]^4 -> [-3, 3]^4`
- `boxes_overlap(b1, b2)`: IoU 기반 overlap 체크
- `box_in_canvas(b)`: canvas 내부 포함 여부
- `norm_to_pixel(b, H=224, W=224)`: 시각화용
- `__main__`: round-trip 변환 확인, 간단한 overlap/bound assertion

### 3. `sampler.py`
- `sample_gt_boxes()` → `(10, 4)` normalized `cx, cy, w, h`
  - size `s ~ Uniform(14, 56)` per class
  - 10개가 서로 non-overlap, canvas 내부 포함될 때까지 재샘플링
  - 일정 시도 횟수 초과 시 fail-safe (예외 또는 재시도)
- `sample_init_signal()` → `(10, 4)` in `[-3, 3]`
  - `z ~ N(0, I_4)` → `clip(z, -3, 3)`
- `__main__`: 한 번 샘플링해 shape, overlap 없음, bound 내부 여부 검증

### 4. `canvas.py`
- `compose_canvas(digits, gt_boxes_pixel)` → `(3, 224, 224)` uint8/float
  - digit을 box size(w, h)로 resize
  - 검은 canvas 위에 paste
  - grayscale digit을 3채널로 복제
- `__main__`: dummy digit 10개 + 랜덤 box로 canvas 생성 → `cv2.imshow`

### 5. `mnist_box_dataset.py` (핵심)
- `class MNISTBoxDataset(Dataset)`
  - `__init__(split, root, image_size=224)`: `split ∈ {train, val, test}`
  - `__len__`: `{50000, 5000, 5000}`
  - `__getitem__(idx)` 절차:
    1. class 0~9 digit 각각 로드
    2. `sample_gt_boxes()` 호출
    3. `compose_canvas()`로 image 합성
    4. `init_signal = sample_init_signal()`
    5. `gt_signal = box_to_signal(gt_boxes)`
    6. `init_boxes = signal_to_box(init_signal)`
    7. dict 반환: `image`, `gt_boxes`, `init_boxes`, `gt_signals`, `init_signals`, `labels`
- 모든 tensor는 `torch.float32` (image만 `[0,1]` 범위)
- `__main__`: sample 1개 뽑아 shape 출력 + `visualize.py` 호출로 `cv2.imshow` / `cv2.waitKey(0)`

### 6. `visualize.py`
- `draw_sample(sample) -> np.ndarray (H, W, 3)`
  - image 위에 gt box (녹색), init box (빨강), digit label 텍스트 overlay
- `show_sample(sample)`: `cv2.imshow` + `cv2.waitKey(0)`
- `__main__`: dataset에서 sample 1개 뽑아 step-by-step 확인

---

## 구현 순서
1. `box_utils.py` (순수 함수, 테스트 쉬움)
2. `sampler.py` (box_utils 사용)
3. `mnist_source.py` (torchvision MNIST 캐시 동작 확인)
4. `canvas.py` (digit + box → image)
5. `visualize.py` (overlay 함수)
6. `mnist_box_dataset.py` (위 모두 조립)

각 단계마다 `python -m dataset.<module>` 로 `__main__` 단독 테스트 후 다음 단계 진행.

---

## 명시적으로 하지 않는 것
- Hungarian matching / classification loss 관련 로직 없음
- trajectory 저장은 training/visualization 단계에서 처리 (dataset 책임 아님)
- data augmentation (flip, rotation 등) 사용하지 않음
- 복잡한 caching / pre-render 없이 `__getitem__`에서 즉석 합성

---

## Sanity 기준
- `dataset[0]` 호출 시:
  - `image.shape == (3, 224, 224)`
  - `gt_boxes.shape == init_boxes.shape == gt_signals.shape == init_signals.shape == (10, 4)`
  - gt box끼리 IoU == 0
  - 모든 gt box가 `[0, 1]` 내부
  - `cv2.imshow`로 숫자 0~9가 각 gt box 위치에 제대로 보임
