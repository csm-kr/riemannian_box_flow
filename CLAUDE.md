# Claude 작업 규칙

## 1. 읽는 순서

항상 아래 순서로 읽는다:

1. `PROJECT.md`
2. `STATUS.md`
3. `TODO.md` (Now 1개만)
4. `ISSUES.md`
5. `plans/active.md`
6. 현재 모듈 plan (예: `plans/dataset.md`)

---

## 2. 핵심 흐름

```
active → module plan → TODO → TDD 구현 → STATUS
```

1. `plans/active.md`에서 방향 정리
2. 확정되면 `plans/{module}.md` 업데이트
3. 실행 단위로 `TODO.md`에 추가
4. TODO Now에서 1개만 선택
5. 선택한 TODO를 TDD 방식으로 구현
6. 완료 후 `STATUS.md` 업데이트

---

## 3. 파일 역할

```
project/
├─ CLAUDE.md             ← Claude 작업 규칙 (지금 이 파일)
├─ PROJECT.md            ← 목표 / 범위 / 성공 기준
├─ STATUS.md             ← 현재 위치
├─ TODO.md               ← 실행 큐
├─ ISSUES.md             ← 문제 / 버그 / 리스크
└─ plans/
   ├─ active.md          ← 전체 방향
   ├─ dataset.md         ← dataset 설계 + 계획
   ├─ model.md           ← model 설계 + 계획
   ├─ training.md        ← training 설계 + 계획
   ├─ visualization.md   ← visualization 설계 + 계획
   └─ archive/           ← 완료된 계획
```

---

## 4. TDD 적용 시점

TDD는 TODO를 실제 구현할 때만 적용한다.

**TDD 하지 않는 단계:**
- `active.md`에서 방향 논의
- `plans/{module}.md` 설계 정리
- `TODO.md` 작업 분해

**TDD 적용 단계:**
- TODO Now의 작업을 실제 코드로 구현할 때

---

## 5. TDD 규칙

1. 실패 테스트 / sanity check 작성
2. 실패 확인
3. 최소 구현
4. 테스트 통과
5. 리팩토링
6. 다시 테스트

---

## 6. 실행 규칙

- 한 번에 TODO 1개만 수행
- 구현 전에 간단한 계획 작성
- 불명확하면 구현하지 말고 질문
- 테스트 없이 완료 처리 금지

---

## 7. 코딩 규칙

- 단순하게 구현
- 모든 모듈 단독 실행 가능
- 가능하면 `__main__`에서 sanity check 실행

---

## 8. 업데이트 규칙

- TODO 완료 → TODO 체크
- 작업 후 → STATUS 업데이트
- 설계 변경 → plans 수정
- 문제 발생 → ISSUES 기록

---

## 9. 금지

- 여러 TODO 동시에 수행 금지
- 테스트 없이 완료 처리 금지
- `archive/` 자동 탐색 금지

---

## 10. Commands

- 빌드: `docker compose up -d --build`
- 접속: `docker exec -it bflow_dev bash`

---

## 11. Git 규칙

- feature 단위로 브랜치 분리: `feature/{module}-{작업명}` (예: `feature/dataset-sampler`)
- 한 feature = 한 브랜치 = 한 PR
- 작업 완료 시 master로 merge 후 브랜치 삭제
- master에 직접 push 금지 (스캐폴드/문서 정리 같은 예외만)

```bash
git checkout -b feature/{module}-{작업명}
# ... 구현 + 커밋
git push -u origin feature/{module}-{작업명}
# PR → review → merge
```

---

## 핵심 요약

설계는 `plans`에서 하고, 실행은 `TODO`에서 하며, TDD는 TODO를 코드로 구현할 때만 적용한다.
