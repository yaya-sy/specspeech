"""Supply ASR transcript blocks to vLLM's native speculative verifier."""

from .draft_buffer import DRAFTS, propose_drafts


def bind_drafter(worker):
    # vLLM's draft callback omits request IDs. Bind its runner once
    # so each request in the batch can read its own SamplingParams.extra_args.
    worker.model_runner.drafter.runner = worker.model_runner
    return worker.model_runner.drafter


class ASRDrafter:
    def __init__(self, vllm_config):
        self.k = vllm_config.speculative_config.num_speculative_tokens
        self.max_model_len = vllm_config.model_config.max_model_len

    def propose(self, new_output_tokens_by_request, context_lengths, token_ids_cpu, *, slot_mappings=None):
        """Return the next ASR draft token block for each request, aligned with verified output.

        Args:
            new_output_tokens_by_request: Newly emitted target token IDs per request,
                including accepted draft tokens and correction or bonus tokens;
                an empty list means this request emitted no tokens during this step.
            context_lengths: Prompt plus verified output token count per request,
                excluding speculative tokens, used to limit proposals to the context capacity.
            token_ids_cpu: CPU token buffer supplied by vLLM; unused here.
            slot_mappings: Optional KV cache slot mappings supplied by vLLM; unused here.
        """
        draft_token_blocks = [[] for _ in new_output_tokens_by_request]
        lookups = []
        for batch_index, new_output_tokens in enumerate(new_output_tokens_by_request):
            if not new_output_tokens:
                # Chunked prefill is still running; do not wait for draft ASR yet.
                continue
            request = self.runner.requests[self.runner.input_batch.req_ids[batch_index]]
            request_args = request.sampling_params.extra_args or {}
            draft_buffer = DRAFTS[request_args["asr_draft_key"]]
            accepted_count = len(new_output_tokens) - 1  # Last token is correction or bonus.
            # Use the actual scheduled block length, including scheduler trimming.
            scheduled_drafts = self.runner.input_batch.spec_token_ids[batch_index]
            draft_buffer.finish_verification(accepted_count, len(scheduled_drafts))
            num_draft_tokens = min(self.k, self.max_model_len - int(context_lengths[batch_index]))
            lookups.append((batch_index, draft_buffer, request.output_token_ids, num_draft_tokens))
        if lookups:
            indices, buffers, outputs, limits = zip(*lookups)
            for index, tokens in zip(indices, propose_drafts(buffers, outputs, limits)):
                draft_token_blocks[index] = tokens
        return draft_token_blocks
