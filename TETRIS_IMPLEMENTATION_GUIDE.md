# TETRIS Implementation Guide

> Reference: "TETRIS: Optimal Draft Token Selection for Batch Speculative Decoding" (ACL 2025)
> Paper: https://arxiv.org/pdf/2502.15197

---

## 1. What TETRIS Does

Batch speculative decoding에서 **고정된 검증 예산(capacity) 내에서 각 시퀀스별 draft prefix 길이를 최적으로 결정**하는 알고리즘.

기존 방식은 배치 내 모든 시퀀스에 동일한 k개의 draft 토큰을 검증한다. TETRIS는 draft 모델의 log-probability를 기반으로 시퀀스별로 검증 길이를 다르게 배분한다. 자신감 높은 prefix에 예산을 더 주고, 불확실한 prefix는 잘라내어 reject 비율을 줄인다.

---

## 2. Required Inputs

TETRIS를 적용하려면 draft 모델의 proposal에 **반드시 아래 3가지가 포함**되어야 한다:

| 텐서 | Shape | 설명 |
|------|-------|------|
| `proposal_token_ids` | `[batch_size, K]` | draft 모델이 제안한 토큰 ID |
| `proposal_logprobs` | `[batch_size, K, vocab_size]` | draft 모델의 **전체 vocab에 대한 log-probability** |
| `proposal_lens` | `[batch_size]` | 각 시퀀스의 유효 proposal 길이 (초기에는 모두 K) |

- `K` = draft에서 생성한 speculative token 수 (= `base_k + extra_proposals`)
- `proposal_logprobs`는 **draft 모델의 logits에 log_softmax를 적용한 값**이다. softmax probability가 아님에 주의.

### `proposal_logprobs`를 얻는 방법

vLLM 구현에서는 `sampler_output.logprobs`로부터 가져온다:

```python
# sampler_output_to_torch() 내부
# shape: [batch_size, proposal_len, vocab_size]
sampled_token_logprobs = torch.stack(
    [sampler_output.logprobs for sampler_output in sampler_output_list],
    dim=0,
)
```

이것은 draft 모델의 forward pass에서 각 step마다 나온 **전체 vocabulary에 대한 log-probability 분포**이다.

---

## 3. Core Algorithm (Exact Implementation)

```python
import torch
from torch_scatter import scatter_max

def select_proposals_no_priority(
    capacity: int,
    proposal_token_ids: torch.Tensor,    # [batch_size, K]
    proposal_logprobs: torch.Tensor,     # [batch_size, K, vocab_size]
    proposal_lens: torch.Tensor,         # [batch_size]
) -> torch.Tensor:
    """
    Returns:
        new_proposal_lens: [batch_size] — TETRIS가 결정한 시퀀스별 검증 길이
    """
    # Step 1: 각 위치에서 "실제 제안된 토큰"의 log-prob만 추출
    #   proposal_logprobs: [B, K, V]
    #   proposal_token_ids: [B, K]
    #   결과 drafts_values: [B, K]
    drafts_values = torch.gather(
        proposal_logprobs,
        dim=-1,
        index=proposal_token_ids.unsqueeze(-1)
    ).squeeze(-1)

    # Step 2: prefix별 cumulative log-probability 계산
    #   위치 t의 값 = log p(tok_1) + log p(tok_2) + ... + log p(tok_t)
    #   이것이 "draft 모델이 이 prefix를 생성할 joint probability의 log"
    drafts_values = drafts_values.cumsum(dim=-1)

    # Step 3: 배치 전체에서 가장 높은 prefix score capacity개 선택
    drafts_values_flatten = drafts_values.flatten()
    _, best_indices = torch.topk(drafts_values_flatten, capacity, largest=True)

    # Step 4: 1D index → (시퀀스 index, prefix 길이) 분해
    seq_indices = best_indices // drafts_values.size(1)    # 어떤 시퀀스
    prefix_lens = (best_indices % drafts_values.size(1)) + 1  # 1-based 길이

    # Step 5: 시퀀스별 최대 prefix 길이 → 새로운 proposal_lens
    new_proposal_lens, _ = scatter_max(
        prefix_lens, seq_indices, dim_size=proposal_lens.size(0)
    )

    return new_proposal_lens
```

### 의존성

- `torch_scatter`의 `scatter_max` 사용 (pip install torch-scatter)
- `scatter_max(src, index, dim_size)`: index가 가리키는 위치별로 src의 max를 계산

---

## 4. Step-by-Step Walkthrough (예시)

```
batch_size = 3, K = 4, capacity = 6

Draft 모델이 제안한 토큰의 log-prob (gather 후):
  seq 0: [-0.10, -0.20, -0.50, -0.70]
  seq 1: [-0.50, -0.70, -0.80, -1.00]
  seq 2: [-0.05, -0.15, -0.10, -0.30]

cumsum 적용 (prefix score):
  seq 0: [-0.10, -0.30, -0.80, -1.50]
  seq 1: [-0.50, -1.20, -2.00, -3.00]
  seq 2: [-0.05, -0.20, -0.30, -0.60]

flatten → 12개 값:
  [-0.10, -0.30, -0.80, -1.50, -0.50, -1.20, -2.00, -3.00, -0.05, -0.20, -0.30, -0.60]

top-6 선택 (largest=True, 값이 0에 가까울수록 좋음):
  index 8  → seq 2, pos 0 → val -0.05
  index 0  → seq 0, pos 0 → val -0.10
  index 9  → seq 2, pos 1 → val -0.20
  index 1  → seq 0, pos 1 → val -0.30
  index 10 → seq 2, pos 2 → val -0.30
  index 4  → seq 1, pos 0 → val -0.50

seq_indices:  [2, 0, 2, 0, 2, 1]
prefix_lens:  [1, 1, 2, 2, 3, 1]   # (pos + 1)

scatter_max 결과:
  seq 0: max(1, 2) = 2
  seq 1: max(1)    = 1
  seq 2: max(1, 2, 3) = 3

최종 proposal_lens: [2, 1, 3]
  → seq 0은 2개, seq 1은 1개, seq 2는 3개 토큰만 검증
  → 총 6개 = capacity ✓ (혹은 그 이하)
```

**핵심 직관**: seq 2가 가장 자신감 높은 prefix를 갖고 있으므로 가장 많은 예산을 받고, seq 1은 불확실하므로 최소한만 받는다.

---

## 5. Capacity 계산

```python
# TETRIS가 켜질 때의 draft 길이
actual_draft_len = base_k + extra_proposals

# capacity = 기본 k × batch_size
capacity = int(base_k * batch_size)
# 즉: capacity = (actual_draft_len - extra_proposals) * batch_size
```

- `base_k`: TETRIS가 없었다면 사용했을 speculative token 수
- `extra_proposals`: TETRIS 전용으로 추가 생성하는 draft 토큰 수
- 전략: **더 넓게 draft (K = base_k + extra)** → **TETRIS가 capacity = base_k × B 만큼만 선별**

예시: `base_k=3, extra_proposals=2, batch_size=8`
- Draft 모델은 5개 토큰 생성
- capacity = 3 × 8 = 24
- TETRIS가 8개 시퀀스에 총 24개 토큰 예산을 최적 배분

---

## 6. Integration into Speculative Decoding Pipeline

```
┌─────────────────────────────────────────────────┐
│ 1. Activation Check                             │
│    tetris_on = batch_size >= min_batch_size      │
│    if tetris_on:                                │
│        draft_len = base_k + extra_proposals     │
│    else:                                         │
│        draft_len = base_k                       │
├─────────────────────────────────────────────────┤
│ 2. Draft Model: generate proposals              │
│    proposal_token_ids  [B, draft_len]           │
│    proposal_logprobs   [B, draft_len, V]        │
│    proposal_lens       [B]  (initially all=K)   │
├─────────────────────────────────────────────────┤
│ 3. ★ TETRIS: select_proposals_no_priority ★      │
│    capacity = base_k * batch_size               │
│    proposal_lens = tetris(capacity, proposals)  │
│    → 각 시퀀스의 proposal_lens가 변경됨          │
├─────────────────────────────────────────────────┤
│ 4. Target Model: score proposals                │
│    target이 draft 토큰을 평가 (proposal_lens     │
│    만큼만 유효하게 처리)                          │
├─────────────────────────────────────────────────┤
│ 5. Verification: accept/reject                  │
│    RejectionSampler 등으로 토큰 수락 결정        │
└─────────────────────────────────────────────────┘
```

### 중요: TETRIS는 `proposal_lens`만 변경한다

- `proposal_token_ids`, `proposal_probs`, `proposal_logprobs` 텐서 자체는 자르지 않는다.
- 오직 `proposal_lens`만 수정하여, 이후 scorer와 verifier가 유효 길이만큼만 처리하도록 한다.
- 이는 scorer/verifier 쪽에서 `proposal_lens`를 올바르게 참조해야 함을 의미한다.

---

## 7. Configuration Parameters

| 파라미터 | 타입 | 설명 |
|----------|------|------|
| `tetris` | bool | TETRIS 활성화 여부 |
| `tetris_extra_proposals` | int | TETRIS용 추가 draft 토큰 수 (논문 실험: 1, 2, 3) |
| `tetris_turn_on_batch_size` | int | TETRIS 활성화 최소 배치 크기 (논문 실험: 2) |

### 비활성 시 fallback 로직

```python
if use_tetris and not tetris_active:
    # 배치가 너무 작아 TETRIS가 꺼졌으면,
    # extra_proposals만큼 draft 길이를 줄여서 일반 SD처럼 동작
    num_lookahead_slots -= tetris_extra_proposals
```

---

## 8. `scatter_max` 없이 구현하기 (순수 PyTorch)

`torch_scatter` 의존성을 피하고 싶다면:

```python
def scatter_max_pure_torch(src, index, dim_size):
    """scatter_max의 순수 PyTorch 대체 구현"""
    out = torch.zeros(dim_size, dtype=src.dtype, device=src.device)
    for i in range(len(src)):
        out[index[i]] = max(out[index[i]], src[i])
    return out

# 또는 벡터화:
def scatter_max_pure_torch_v2(src, index, dim_size):
    out = torch.zeros(dim_size, dtype=src.dtype, device=src.device)
    out.scatter_reduce_(0, index, src, reduce="amax")
    return out
```

> Note: `torch.Tensor.scatter_reduce_` (PyTorch 2.0+) 사용 시 `reduce="amax"`로 동일 동작.
> 단, top-k에 선택되지 않은 시퀀스는 `proposal_lens=0`이 됨에 주의 (검증 자체를 건너뜀).

---

## 9. Complete Self-Contained Reference Implementation

```python
import torch

def tetris_select(
    proposal_token_ids: torch.Tensor,    # [B, K]
    proposal_logprobs: torch.Tensor,     # [B, K, V]
    proposal_lens: torch.Tensor,         # [B]
    base_k: int,
    extra_proposals: int,
    min_batch_size: int = 2,
) -> torch.Tensor:
    """
    TETRIS: 배치 내 시퀀스별 최적 draft prefix 길이를 결정.
    
    Args:
        proposal_token_ids: draft 모델이 제안한 토큰 ID
        proposal_logprobs: draft 모델의 전체 vocab log-probability
        proposal_lens: 현재 각 시퀀스의 proposal 길이
        base_k: TETRIS가 없을 때의 기본 speculative token 수
        extra_proposals: TETRIS용 추가 draft 토큰 수
        min_batch_size: TETRIS 활성화 최소 배치 크기
    
    Returns:
        new_proposal_lens: [B] — 각 시퀀스의 새로운 검증 길이
    """
    B = proposal_token_ids.size(0)
    
    if B < min_batch_size:
        return proposal_lens
    
    capacity = base_k * B
    
    # (1) 제안된 토큰의 log-prob 추출: [B, K]
    token_logprobs = torch.gather(
        proposal_logprobs, dim=-1,
        index=proposal_token_ids.unsqueeze(-1)
    ).squeeze(-1)
    
    # (2) prefix cumulative log-prob: [B, K]
    prefix_scores = token_logprobs.cumsum(dim=-1)
    
    # (3) global top-capacity 선택
    flat_scores = prefix_scores.flatten()
    
    actual_capacity = min(capacity, flat_scores.numel())
    _, top_indices = torch.topk(flat_scores, actual_capacity, largest=True)
    
    # (4) index 분해
    K = prefix_scores.size(1)
    seq_indices = top_indices // K
    prefix_lens = (top_indices % K) + 1  # 1-based
    
    # (5) 시퀀스별 최대 prefix 길이
    new_lens = torch.zeros(B, dtype=prefix_lens.dtype, device=prefix_lens.device)
    new_lens.scatter_reduce_(0, seq_indices, prefix_lens, reduce="amax")
    
    return new_lens
```

---

## 10. Common Pitfalls

### 10-1. `proposal_logprobs`는 전체 vocab에 대한 log-prob이어야 한다

- shape은 `[B, K, vocab_size]`
- gather로 특정 토큰의 log-prob만 뽑아쓰지만, 입력 자체는 full vocab 분포
- 만약 draft 모델에서 sampled token의 log-prob만 저장한다면 (`[B, K]` 스칼라), 바로 Step 1의 결과로 사용 가능. 단, 이 경우 gather 단계를 건너뜀

### 10-2. cumsum은 dim=-1 (proposal 길이 축)

- `dim=0`(배치 축)이 아님에 주의
- **같은 시퀀스 내에서** 위치 1부터 t까지의 누적합

### 10-3. `capacity` 계산에서 extra_proposals를 빼야 한다

```python
# 올바른 계산:
capacity = (proposal_len - extra_proposals) * batch_size

# 틀린 계산 (예산 초과):
capacity = proposal_len * batch_size
```

### 10-4. top-k에 선택되지 않은 시퀀스는 `proposal_lens = 0`

- scatter_max 초기값이 0이므로, top-k에 한 번도 포함되지 않은 시퀀스는 검증을 건너뜀
- 이는 의도된 동작: 해당 시퀀스는 draft가 매우 불확실하므로 예산을 쓰지 않는 것이 이득

### 10-5. TETRIS는 `proposal_lens`만 수정한다

- `proposal_token_ids`, `proposal_probs` 등의 텐서는 건드리지 않음
- 이후 scorer/verifier가 `proposal_lens`를 참조하여 유효 범위만 처리해야 함

---

## 11. Mathematical Formulation

배치 내 시퀀스 집합 \( \mathcal{B} = \{1, \ldots, B\} \), 시퀀스 \(i\)의 draft prefix 길이 \(l_i\), 위치 \(t\)의 draft log-prob \(\log p_d(x_t^{(i)})\)라 하면:

**Prefix score:**
$$s(i, t) = \sum_{j=1}^{t} \log p_d(x_j^{(i)})$$

**TETRIS 최적화 문제 (relaxed):**
$$\max \sum_{(i, t) \in \mathcal{S}} s(i, t) \quad \text{s.t.} \quad |\mathcal{S}| \leq C$$

여기서 \( C = k_{\text{base}} \times B \) (capacity), \(\mathcal{S}\)는 선택된 (시퀀스, 위치) 쌍의 집합.

실제 구현에서는 global top-C를 뽑은 뒤, **시퀀스별로 가장 긴 prefix를 채택**하므로:
$$l_i = \max_{(i, t) \in \mathcal{S}} t$$

이는 prefix monotonicity (위치 1~t를 모두 검증해야 위치 t를 검증 가능)를 반영한 것이다.

---

## 12. Experiment Configuration Reference

논문 실험에서의 sweep 범위:

```
base_k:          1, 2, 3, 4, 5, 6
extra_proposals: 1, 2, 3
min_batch_size:  2

Target models:
  - vicuna-33b-v1.3       (draft: vicuna-68m,        TP=4)
  - Llama-3.1-70B         (draft: Llama-3.2-1B-FP8,  TP=8)
  - Llama-3.1-405B-FP8    (draft: Llama-3.2-1B-FP8,  TP=8)

Datasets: ShareGPT, Arena, Domain-Tough
Requests per run: 512
Repeat: 3 runs per config
```
