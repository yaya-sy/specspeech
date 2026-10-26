# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Batched GPU suffix matching against external ASR drafts.

Adapted from vLLM 0.30's NgramGPUKernel (unfold, parallel matches, argmax):
https://github.com/vllm-project/vllm/blob/v0.30.0/vllm/v1/spec_decode/ngram_proposer_gpu.py
Changes: external drafts, normalized token IDs, bounded cursors, and empty-token
skipping. See VLLM_LICENSE for the original license.
"""

import torch
from torch import nn


def _compact(tokens):
    """Pack nonempty normalized IDs into columns 1..N; column zero is padding."""
    counts = (tokens != 0).long().cumsum(dim=1)
    packed = tokens.new_zeros((tokens.shape[0], tokens.shape[1] + 1))
    packed.scatter_(1, torch.where(tokens != 0, counts, 0), tokens)
    return packed, counts


class SuffixGPUKernel(nn.Module):
    def __init__(self, normalized_ids, window_widths=(8,)):
        super().__init__()
        self.register_buffer("normalized_ids", normalized_ids)
        self.window_widths = tuple(window_widths)

    def forward(self, output_tokens, draft_tokens, bounds):
        """Return draft cursors [batch] on GPU, with -1 for no match.

        output_tokens: Last 16 target IDs per request, padded with -1 [batch, 16].
        draft_tokens: Raw IDs within the largest window, padded with -1 [batch, 2 * max_width + 1].
        bounds: Largest-window start, exclusive stop, last evaluated token, earliest proposal [batch, 4].
        """
        output, output_counts = _compact(self.normalized_ids[output_tokens + 1])
        normalized_draft = self.normalized_ids[draft_tokens + 1]
        raw_positions = bounds[:, :1] + torch.arange(draft_tokens.shape[1], device=draft_tokens.device)
        cursors = bounds.new_full((bounds.shape[0],), -1)
        for window in self.window_widths:
            draft = torch.where((raw_positions >= bounds[:, 2:3] - window)
                                & (raw_positions < bounds[:, 2:3] + window + 1), normalized_draft, 0)
            packed, _ = _compact(draft)
            candidates = bounds[:, 3:4] + torch.arange(window + 2, device=draft.device)
            counts = ((raw_positions[:, None, :] < candidates[:, :, None])
                      & (draft[:, None, :] != 0)).sum(dim=-1)
            matches_by_length = []
            for length in (1, 2, 3):
                search_windows = packed[:, 1:].unfold(1, length, 1)
                starts = (counts - length).clamp(0, search_windows.shape[1] - 1)
                candidate_ngrams = search_windows.gather(
                    1, starts[:, :, None].expand(-1, -1, length),
                )
                suffix_indices = output_counts[:, -1:] - length + 1 + torch.arange(length, device=draft.device)
                suffix = output.gather(1, suffix_indices.clamp(min=0))
                matches = (candidate_ngrams == suffix[:, None, :]).all(dim=-1)
                matches &= (counts >= length) & (output_counts[:, -1:] >= length)
                matches &= candidates <= torch.minimum(bounds[:, 1:2], bounds[:, 2:3] + window + 1)
                first_match = matches.int().argmax(dim=1)
                has_match = matches.gather(1, first_match[:, None]).squeeze(1)
                position = candidates.gather(1, first_match[:, None]).squeeze(1)
                matches_by_length.append(torch.where(has_match, position, -1))
            positions = torch.stack(matches_by_length, dim=1)
            best = 2 - (positions >= 0).int().flip(dims=[1]).argmax(dim=1)
            position = positions.gather(1, best[:, None]).squeeze(1)
            # Keep the first matching window per request, without a CPU round trip.
            cursors = torch.where(cursors >= 0, cursors, position)
        return cursors


class SuffixGPULookup:
    def __init__(self, tokenizer, vocab_size, device, window_widths=(8,)):
        self.device, self.window = device, max(window_widths)
        # Build once per engine, outside timed inference. Equal normalized text
        # gets the same integer; zero represents punctuation, special IDs or padding.
        symbols, normalized_ids = {"": 0}, [0]
        texts = tokenizer.batch_decode([[token] for token in range(vocab_size)],
                                      skip_special_tokens=True, clean_up_tokenization_spaces=False)
        for decoded in texts:
            text = "".join(char for char in decoded.casefold() if char.isalnum())
            normalized_ids.append(symbols.setdefault(text, len(symbols)))
        kernel = SuffixGPUKernel(torch.tensor(normalized_ids, device=device), window_widths)
        self.kernel = torch.compile(kernel, fullgraph=True, dynamic=True)

    def __call__(self, outputs, drafts, last_evaluated, min_positions):
        """Return CPU cursors for a batch; the synchronous verifier needs this handoff."""
        width = 2 * self.window + 1
        bounds = [(max(0, position - self.window),
                   min(len(draft), position + self.window + 1), position, max(position, minimum))
                  for draft, position, minimum in zip(drafts, last_evaluated, min_positions)]
        windows = [draft[start:stop] for draft, (start, stop, _, _) in zip(drafts, bounds)]
        tails = [output[-16:] for output in outputs]
        cursors = self.kernel(
            torch.tensor([tail + [-1] * (16 - len(tail)) for tail in tails], device=self.device),
            torch.tensor([tokens + [-1] * (width - len(tokens)) for tokens in windows], device=self.device),
            torch.tensor(bounds, device=self.device),
        ).tolist()
        return [position if position >= 0 else None for position in cursors]
