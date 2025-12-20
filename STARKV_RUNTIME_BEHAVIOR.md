# StarKV 런타임 동작 요약 (코드 기반)

이 문서는 **이 레포의 현재 코드 상태**에서 StarKV가 실제로 어떻게 동작하는지(무엇이 구현되어 있고 무엇이 placeholder인지)를 정리합니다.  
목표는 “실험이 왜 이런 결과가 나오는지”를 빠르게 추적하고, 디버깅 포인트를 명확히 하는 것입니다.

## 1) 큰 그림: 어떤 컴포넌트가 어디서 작동하나

- **StarKV 옵션 플래그(Entry/Config)**  
  - `--enable-starkv-super-cache`: StarKV 플러그인 경로 활성화  
  - `--starkv-confidence-threshold`: confidence 기반 reforward 임계값  
  - `--starkv-compression-ratio`, `--starkv-score-fn`, `--starkv-offload` 등
  - 정의/파싱: `vllm-sampling/vllm/config/cache.py`, `vllm-sampling/vllm/engine/arg_utils.py`

- **SuperPress(점수/의사결정 엔진) 어댑터**: `vllm-sampling/vllm/starkv/adapter.py`
  - 외부 `starkv` 패키지를 우선 import 시도
  - 없으면 내부 placeholder `vllm.starkv.superpress.SuperPress`로 fallback

- **Worker(실제 실행 위치)**: `vllm-sampling/vllm/v1/worker/gpu_model_runner.py`
  - prefill 단계에서 snapshot 생성 → adapter 호출
  - decode 단계에서 logits 기반 confidence 계산 → reforward plan 생성(추가 구현)
  - offload(옵션) 활성 시 host buffer에 KV block 미러링

- **Scheduler/CacheManager(계획 반영)**  
  - `StarKVCacheManager`: `vllm-sampling/vllm/v1/core/starkv_cache_manager.py`
  - scheduler: `vllm-sampling/vllm/v1/core/sched/scheduler.py`
  - Worker가 남긴 `starkv_feedback`를 스케줄러가 ingestion → reforward 플래그/plan 저장

---

## 2) SuperPress 로딩/구성(외부 vs 내부)

### 2.1 로딩 우선순위

`StarKVPressAdapter`는 다음 순서로 SuperPress를 준비합니다.

1. 외부 패키지 우선:
   - `from starkv import SuperPress`
2. 실패 시 내부 구현 fallback:
   - `from vllm.starkv.superpress import SuperPress`
3. 둘 다 실패 시 StarKV 비활성화(경고 로그)

코드: `vllm-sampling/vllm/starkv/adapter.py`

### 2.2 CacheConfig → SuperPress 설정 전달

adapter는 아래 필드를 press 인스턴스에 주입합니다(해당 attribute가 존재할 때).

- `starkv_compression_ratio` → `press.compression_ratio`
- `starkv_score_fn` → `press.score_fn`
- `starkv_confidence_threshold` → `press.confidence_threshold`

코드: `vllm-sampling/vllm/starkv/adapter.py`

---

## 3) Prefill 단계 동작: snapshot → score_prefill()

### 3.1 언제 호출되나

prefill 입력을 준비할 때(attention metadata를 만드는 루프에서) layer별로 `_notify_starkv_prefill()`이 호출됩니다.

코드: `vllm-sampling/vllm/v1/worker/gpu_model_runner.py`

### 3.2 어떤 데이터를 전달하나

worker는 아래 형태의 snapshot을 만들어 adapter에 전달합니다.

- `layer_name`, `kv_cache_group_id`
- `request_ids`
- `num_tokens` (이번 스텝에 스케줄된 토큰 수)
- `slot_mapping` (CPU로 복사된 slot mapping)
- `token_request_indices`, `token_positions`

snapshot 타입: `StarKVLayerSnapshot`  
코드: `vllm-sampling/vllm/starkv/adapter.py`

### 3.3 adapter에서 호출하는 API

adapter는 `press.score_prefill(payload)`를 호출하고, 결과를 `StarKVPrefillResult`로 파싱합니다.

기대 결과(딕셔너리):
- `confidence_by_request`: `{req_id: float}`
- (선택) `low_confidence_requests`: `[req_id, ...]`
- (선택) `keep_tokens_by_request`: `{req_id: [token_idx...]}` (⚠️ 현재 placeholder path에서 위험할 수 있음)

코드: `vllm-sampling/vllm/starkv/adapter.py`

### 3.4 “reforward 직후에는 confidence 재검사 스킵” 가드

prefill에서도, reforward 직후 1 step 동안 `low_confidence_requests`를 제거해서 연속 reforward 트리거를 완화합니다.

코드: `vllm-sampling/vllm/v1/worker/gpu_model_runner.py` (`_notify_starkv_prefill`)

---

## 4) Decode 단계 동작(추가 구현): logits → top-1 softmax confidence

### 4.1 언제 계산되나

decode 스텝마다 `execute_model()`에서
1) forward + logits 계산  
2) sampling  
3) bookkeeping 직후  
에 “decode-confidence reforward” 체크가 수행됩니다.

코드: `vllm-sampling/vllm/v1/worker/gpu_model_runner.py`

### 4.2 confidence 정의

각 request에 대해 logits \( \ell \in \mathbb{R}^V \)에서:

\[
p_{top1} = \exp(\max(\ell) - \log\sum\exp(\ell))
\]

즉, **output distribution의 top-1 softmax probability**를 confidence로 사용합니다.

### 4.3 트리거 조건

- `p_top1 <= starkv_confidence_threshold` 이면 reforward 후보
- 임계값이 `>= 1.0`이면 트리거 불가능(즉시 return)

### 4.4 “마지막 reforward 이후 decoded tokens 전체를 reforward”

reference 코드의 `last_superkv_index`를 worker에서 request별로 추적합니다.

- `self._starkv_last_reforward_index[req_id]` (absolute token position; prompt 포함)
- 초기값은 `prompt_len`로 간주하여 **prompt 영역으로 rewind하지 않게** 합니다.

low-confidence 발생 시 plan:
- `start_pos = last_reforward_index`
- `end_pos = current_pos`
- `suffix_len = end_pos - start_pos`
- 이후 `last_reforward_index = end_pos`로 갱신

### 4.5 디버그 로그

reforward 트리거 시 아래가 출력됩니다:
- `req_id`, `conf`, `thr`, `start_pos`, `end_pos`, `suffix_len`

INFO로 보고 싶으면:

```bash
export STARKV_LOG_REFORWARD=1
```

코드: `vllm-sampling/vllm/v1/worker/gpu_model_runner.py`

---

## 5) Scheduler로 reforward가 전달되는 방식(플로우)

1. Worker가 `starkv_feedback`에 `StarKVLayerFeedback`을 append
2. Scheduler가 output을 수집할 때 `kv_cache_manager.ingest_layer_feedbacks(starkv_feedback)` 호출
3. `StarKVCacheManager`가 `metadata["starkv_reforward_requests"]`를 `_reforward_plans`에 기록
4. Scheduler scheduling 단계에서 `consume_reforward_flag(req_id)`가 True면 `pop_reforward_plan(req_id)`를 꺼내 rewind/suffix_len 적용

코드:
- `vllm-sampling/vllm/v1/core/starkv_cache_manager.py`
- `vllm-sampling/vllm/v1/core/sched/scheduler.py`

---

## 6) Offload (옵션) 동작과 주의점

### 6.1 동작 개요

`--starkv-offload`가 켜져 있고, keep-plan이 생성되어 block drop이 일어나면,
worker의 `StarKVOffloadStore`가 GPU KV cache의 특정 block을 CPU host buffer로 복사합니다.

코드:
- `vllm-sampling/vllm/starkv/offload.py`
- worker에서 호출: `vllm-sampling/vllm/v1/worker/gpu_model_runner.py`

### 6.2 중요한 버그/수정 사항

KV cache tensor shape이 레이어/백엔드에 따라 다음처럼 달라질 수 있습니다:
- `(num_blocks, ...)`
- `(2, num_blocks, ...)` (K/V가 dim0)

따라서 offload/restore는 block dim을 자동 판별해서 `index_select/index_copy_`를 해야 하며,
인덱스 범위 체크 없이 실행하면 CUDA device-side assert가 날 수 있습니다.

현재 코드에서는:
- block dim 자동 선택
- out-of-range 인덱스 제거 후 처리

코드: `vllm-sampling/vllm/starkv/offload.py`

---

## 7) 현재 구현의 한계(중요)

- **“SuperCache tier 전환” 같은 진짜 StarKV 캐시 계층 구현은 아직 없음**  
  현재 reforward는 cache tier가 아니라, scheduler의 **rewind + replay**로 근사합니다.

- **keep_tokens_by_request 기반 “block drop/keep”는 placeholder이며 위험할 수 있음**  
  실제 KV/block 매핑을 정확히 아는 SuperPress가 아니면, 잘못된 block drop/offload로 불안정해질 수 있습니다.

- **threshold가 매우 높을 때(예: 0.99)** decode-confidence 트리거가 매우 자주 발생할 수 있으며,
  이 경우 마지막 일부 request에서 progress가 매우 느려 “hang처럼 보이는” 상황이 발생할 수 있습니다.

---

## 8) 빠른 실행 예시(bench throughput)

```bash
python -m vllm.entrypoints.cli.main bench throughput \
  --backend vllm \
  --model meta-llama/Meta-Llama-3.1-8B-Instruct \
  --load-format dummy \
  --dataset-name random \
  --num-prompts 32 \
  --input-len 1024 \
  --output-len 64 \
  --enforce-eager \
  --enable-starkv-super-cache \
  --starkv-confidence-threshold 0.99 \
  --output-json starkv.json
```

INFO 로그로 reforward 트리거를 보려면:

```bash
export STARKV_LOG_REFORWARD=1
```


