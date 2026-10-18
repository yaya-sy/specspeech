"""Feed audio requests to vLLM while their ASR drafts run in the background."""

from collections import deque
from dataclasses import dataclass, field
from itertools import islice
from time import perf_counter
from uuid import uuid4

import torch
from vllm import SamplingParams
from vllm.sampling_params import RequestOutputKind

from .draft_buffer import DRAFTS, DRAFT_CHANGED, DraftBuffer


@dataclass
class Request:
    sample: dict
    audio: object
    prepared: object
    draft: object = None
    parts: list = field(default_factory=list)
    output: object = None
    started: float = field(default_factory=perf_counter)


def stream_inference(verifier, samples, prepare, *, max_pending, max_tokens,
                     drafter=None, punctuator=None):
    """Yield (metadata, result) pairs; vLLM owns scheduling and KV allocation.

    samples: Iterator of (metadata, mono 16 kHz waveform) pairs.
    prepare: Convert a waveform into a vLLM prompt.
    max_pending: Bound admitted requests; refill after individual completions.
    drafter: Optional ASR engine publishing growing draft tokens.
    """
    if max_pending <= 0:
        raise ValueError("max_pending must be positive")
    engine = verifier.engine.llm_engine
    samples = iter(samples)
    queued = deque()
    pending = {}
    input_done = stopped = False
    input_error = None
    batch_size = drafter.batch_size if drafter else verifier.max_num_seqs

    def update_drafts(requests):
        for batch in drafter.stream_batch(
            [request.audio for request in requests],
            is_finished=lambda index: requests[index].output is not None,
        ):
            if stopped:
                return
            texts = [text for _, text, _ in batch]
            if punctuator is not None:
                texts = punctuator(texts)
            for (index, _, last), text in zip(batch, texts):
                request = requests[index]
                if request.output is not None:
                    continue
                piece = (" " if request.parts else "") + text
                request.parts.append(text)
                request.draft.append(verifier.encode(piece), finished=last)

    def prepare_inputs():
        nonlocal input_done, input_error
        requests = []
        try:
            while not stopped and (inputs := list(islice(samples, batch_size))):
                requests = [Request(
                    sample=sample,
                    audio=audio,
                    prepared=engine.renderer.render_cmpl([prepare(audio)])[0],
                    draft=DraftBuffer(verifier.decode, lookup=verifier.lookup) if drafter else None,
                ) for sample, audio in inputs]
                with DRAFT_CHANGED:
                    DRAFT_CHANGED.wait_for(lambda: stopped or not queued)
                    if stopped:
                        return
                    queued.extend(requests)
                    DRAFT_CHANGED.notify_all()
                if drafter:
                    update_drafts(requests)
        except Exception as error:
            input_error = error
            for request in requests:
                if request.draft is not None:
                    request.draft.fail(error)
        finally:
            with DRAFT_CHANGED:
                input_done = True
                DRAFT_CHANGED.notify_all()

    def ready_requests():
        for _ in range(max_pending - len(pending)):
            with DRAFT_CHANGED:
                if not pending:
                    DRAFT_CHANGED.wait_for(lambda: queued or input_done)
                if input_error is not None:
                    raise input_error
                if not queued:
                    return
                request = queued.popleft()
                DRAFT_CHANGED.notify_all()
            yield request

    input_job = verifier.input_executor.submit(prepare_inputs)
    try:
        with torch.cuda.stream(verifier.stream):
            while True:
                for request in ready_requests():
                    request_id = uuid4().hex
                    pending[request_id] = request
                    params = SamplingParams(temperature=0, max_tokens=max_tokens,
                                            output_kind=RequestOutputKind.FINAL_ONLY)
                    if drafter:
                        DRAFTS[request_id] = request.draft
                        params.extra_args = {"asr_draft_key": request_id}
                    engine.add_request(request_id, request.prepared, params)

                if not pending:
                    break
                if engine.has_unfinished_requests():
                    for output in engine.step():
                        if output.finished:
                            pending[output.request_id].output = output.outputs[0]
                for request_id, request in list(pending.items()):
                    if request.draft is not None and request.draft.error is not None:
                        raise request.draft.error
                    if request.output is None:
                        continue
                    result = speculative_result(verifier, request)
                    if drafter:
                        DRAFTS.pop(request_id, None)
                    del pending[request_id]
                    yield request.sample, result
    finally:
        with DRAFT_CHANGED:
            stopped = True
            DRAFT_CHANGED.notify_all()
        input_job.result()
        if pending:
            engine.abort_request(list(pending))
        if drafter:
            for request_id in pending:
                DRAFTS.pop(request_id, None)


def speculative_result(verifier, request):
    """Format a verified transcript and vLLM's acceptance statistics."""
    output = request.output
    metrics = output.spec_decode_metrics
    histogram = metrics.histogram if metrics else []
    stats = metrics.to_dict() if metrics else {}
    proposed = stats.get("num_draft_tokens", 0)
    accepted = sum(length * count for length, count in enumerate(histogram))
    return {
        "draft": " ".join(request.parts),
        "text": verifier.decode(output.token_ids),
        "tokens": len(output.token_ids),
        "finish_reason": output.finish_reason,
        "request_seconds": perf_counter() - request.started,
        "draft_tokens": len(request.draft.tokens) if request.draft else 0,
        "calls": sum(histogram),
        "proposed": proposed,
        "accepted": accepted,
        "rejected": proposed - accepted,
        "spec_decode_metrics": stats,
    }
