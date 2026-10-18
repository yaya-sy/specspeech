"""Hold ASR decode requests when their incremental draft runs out.

Uses vLLM 0.30's scheduler hook; prefill, KV allocation and rollback stay native.
"""

from vllm.v1.core.sched.scheduler import Scheduler

from .draft_buffer import DRAFTS, DRAFT_CHANGED, propose_drafts


def request_draft(request):
    args = request.sampling_params.extra_args or {}
    return DRAFTS.get(args.get("asr_draft_key"))


class ASRScheduler(Scheduler):
    def schedule(self, *args, **kwargs):
        # Refresh proposals before budgeting tokens, including newly arrived pieces.
        requests = [request for request in self.running if request_draft(request) is not None]
        proposals = propose_drafts(
            [request_draft(request) for request in requests], [request.output_token_ids for request in requests],
            [min(self.num_spec_tokens, self.max_model_len - request.num_tokens) for request in requests],
        )
        for request, tokens in zip(requests, proposals):
            request.spec_token_ids = tokens
        output = super().schedule(*args, **kwargs)
        if output.total_num_scheduled_tokens == 0:
            paused = [draft for request in self.running
                      if (draft := request_draft(request)) is not None and draft.waiting]
            if paused:
                # Sleep only when no request can run; publication also wakes errors.
                with DRAFT_CHANGED:
                    DRAFT_CHANGED.wait_for(lambda: any(not draft.waiting for draft in paused))
        return output

    def _reserve_prefill_lookahead(self, request, num_computed_tokens, num_new_tokens):
        draft = request_draft(request)
        if num_computed_tokens >= request.num_prompt_tokens and draft is not None and draft.waiting:
            return 0
        return super()._reserve_prefill_lookahead(request, num_computed_tokens, num_new_tokens)
