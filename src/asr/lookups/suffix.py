"""Find the next draft position from normalized verified token n-grams."""


def suffix_lookup(output, draft_tokens, last_evaluated, normalize, window_widths=(8,), *, min_position=0):
    """Match a target suffix near the verification stopping position, or return None.

    Args:
        output: Verified target token IDs ending at the current output position.
        draft_tokens: Draft transcript token IDs in the verifier's vocabulary.
        last_evaluated: Index of the rejected draft token, or the last token if all were accepted.
        normalize: Convert a token ID to lowercase alphanumeric text.
        window_widths: Tokens on either side of last_evaluated, tried in order until a match.
        min_position: Earliest proposal position, excluding already accepted draft tokens.
    """
    target_suffix = [piece for token in output[-16:] if (piece := normalize(token))]
    for window in window_widths:
        start = max(0, last_evaluated - window)
        stop = min(len(draft_tokens), last_evaluated + window + 1)
        draft = [normalize(token) for token in draft_tokens[start:stop]]
        for length in (3, 2, 1):
            if len(target_suffix) < length:
                continue
            for position in range(max(last_evaluated, min_position), stop + 1):
                prefix = [piece for piece in draft[:position - start] if piece]
                if prefix[-length:] == target_suffix[-length:]:
                    return position
    return None
