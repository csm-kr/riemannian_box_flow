# dataset.md

## Dataset Name
MNIST-10 Box Flow Toy Dataset

## Objective
검은색 224x224 canvas 위에 MNIST 숫자 0~9를 각각 하나씩 배치하여,  
init box state에서 gt box state로 이동하는 box flow toy dataset을 만든다.

---

## Sample Structure
각 sample은 아래를 포함한다.
- `image`: `(3, 224, 224)` black canvas 기반 composite image
- `gt_boxes`: `(10, 4)` in normalized `cx, cy, w, h`
- `init_boxes`: `(10, 4)` in normalized `cx, cy, w, h`
- `gt_signals`: `(10, 4)` in `[-3, 3]`
- `init_signals`: `(10, 4)` in `[-3, 3]`
- `labels`: optional, `[0,1,2,3,4,5,6,7,8,9]`

---

## Image Composition
- canvas size: `224 x 224`
- image shape: `(3, 224, 224)`
- background: black
- MNIST digit은 grayscale로 생성하되 3채널로 복제하여 사용
- 각 image에는 숫자 0~9가 각각 1개씩 존재
- class 중복 없음
- gt box 간 overlap 없음
- 모든 gt box는 canvas 내부에 완전히 포함

---

## Box Size
- gt box size는 매 sample마다 랜덤
- size range: `14 ~ 56` pixels
- 기본은 square box 사용: `w = h = s`, `s ~ Uniform(14, 56)`

---

## Init Boxes
- init box는 총 10개 사용한다
- 각 init box는 숫자 0~9 클래스에 1:1 대응한다
- init box는 이미지 내부 valid box일 필요는 없다
- 생성 순서:
  1. `init_signals`: `z ~ N(0, I_4)` → `clip(z, -3, 3)` 로 샘플링
  2. `init_boxes`: `init_signals`를 `b = (s/3 + 1) / 2` 로 변환하여 `[0,1]^4` 로 저장
- 두 값 모두 sample dict에 항상 저장한다

---

## GT Boxes
- 각 sample마다 class별 GT center를 랜덤 샘플링
- GT box size도 sample마다 랜덤 샘플링
- 10개 GT box는 서로 겹치지 않아야 함
- overlap 또는 out-of-bound면 재샘플링

---

## Signal Space
공통 signal space는 `[-3, 3]^4` 를 사용한다.

### Init Signal
초기 signal은 아래처럼 샘플링한다.

\[
z \sim \mathcal{N}(0, I_4)
\]

\[
\tilde{b}_0 = \mathrm{clip}(z, -3, 3)
\]

### GT Signal
GT box \(b_1 \in [0,1]^4\) 를 같은 signal space로 변환한다.

\[
\tilde{b}_1 = (2b_1 - 1)\cdot 3
\]

여기서 \(b_1\)은 normalized GT box in `[0,1]^4` 이다.

### Signal to Box
필요 시 signal을 box로 되돌릴 때는 아래 변환을 사용한다.

\[
b = \frac{\tilde{b}/3 + 1}{2}
\]

---

## Recommended Sampling Procedure
1. 숫자 0~9 각각에 대해 MNIST digit 하나 샘플링
2. 각 class의 gt box size를 랜덤 샘플링
3. non-overlap 조건으로 gt center 샘플링
4. digit을 resize 후 black canvas에 paste
5. gt box를 normalized `cx, cy, w, h`로 저장
6. init signal을 `clip(N(0,1), -3, 3)`으로 생성
7. gt box를 `[-3, 3]` signal로 변환
8. 필요 시 init signal을 box로 변환하여 `init_boxes` 저장

---

## Labels
- 이 toy에서는 query index가 class identity 역할을 한다
- query 0은 digit 0, query 1은 digit 1, ..., query 9는 digit 9를 담당한다
- 따라서 초기 버전에서는 classification loss가 필요 없다
- `labels`는 optional metadata로만 저장한다

---

## Visualization
- dataset debug visualization은 OpenCV 기반으로 제공한다
- image 위에 gt box와 init box를 함께 그려서 확인할 수 있어야 한다
- visualization은 `cv2.imshow()`와 `cv2.waitKey()`로 sample을 step-by-step 확인할 수 있어야 한다
- digit label, init box, gt box를 색상으로 구분해서 표시한다

---

## Evaluation Hooks
- 각 index별 start box `b0[i]` 와 target box `b1[i]` 를 저장한다
- ODE trajectory 시각화를 위해 intermediate boxes `b_t[i]` 를 저장할 수 있어야 한다
- trajectory sampling은 `t = 0.0, 0.1, 0.2, ..., 1.0` 기준으로 한다
- 각 index `i` 에 대해 `b0[i] -> ... -> b1[i]` 과정을 GIF로 export 할 수 있어야 한다
- GIF에는 현재 box, start box, target box를 image 위에 overlay 해서 보여준다
- optional: 10개 index를 한 화면에 함께 보여주는 통합 GIF도 지원한다
- 통합 GIF 역시 ODE timestep `0.1` 간격으로 생성한다

---

## Splits
- train: 50,000
- val: 5,000
- test: 5,000

---

## Outputs
권장 저장 항목:
- `images`
- `gt_boxes`
- `init_boxes`
- `gt_signals`
- `init_signals`
- `labels` (optional)

optional debug/eval outputs:
- `trajectory_boxes`
- `trajectory_gif_per_index`
- `trajectory_gif_combined`# plan.md — Dataset 구현 계획

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
