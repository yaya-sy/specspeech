"""Growing ASR drafts shared by the in-process vLLM scheduler and worker."""

from threading import Condition

from ...lookups.suffix import suffix_lookup


DRAFTS = {}
DRAFT_CHANGED = Condition()

class DraftBuffer:
    def __init__(self, decode, tokens=(), finished=False, *, lookup=None, lookup_window_width=(8,)):
        self.decode = decode
        self.lookup = lookup
        self.lookup_window_width = lookup_window_width
        self.tokens = list(tokens)
        self.finished = finished
        self.error = None
        self.cursor = self.last_accepted = self.last_evaluated = 0
        self._normalized = {}
        self._lookup_state = None

    def append(self, tokens, *, finished=False):
        with DRAFT_CHANGED:
            # Published tokens never change when the next audio piece arrives.
            self.tokens.extend(tokens)
            self.finished = finished
            DRAFT_CHANGED.notify_all()

    def fail(self, error):
        with DRAFT_CHANGED:
            self.error = error
            DRAFT_CHANGED.notify_all()

    @property
    def waiting(self):
        with DRAFT_CHANGED:
            if self.error is not None:
                raise self.error
            return not self.finished and (
                self.last_accepted >= len(self.tokens) or self.cursor == len(self.tokens)
            )

    def finish_verification(self, accepted_count, draft_count):
        """Record the stopping token and accepted prefix of the scheduled draft block."""
        with DRAFT_CHANGED:
            if draft_count:
                self.last_evaluated = self.cursor + min(accepted_count, draft_count - 1)
                if accepted_count:
                    self.last_accepted = self.cursor + accepted_count

    def propose(self, output, k):
        return propose_drafts([self], [output], [k])[0]

    def _state(self, output):
        return len(output), len(self.tokens), self.last_evaluated, self.last_accepted

    def _text(self, token):
        if token not in self._normalized:
            self._normalized[token] = "".join(
                char for char in self.decode([token]).casefold() if char.isalnum()
            )
        return self._normalized[token]


def propose_drafts(buffers, outputs, limits):
    """Batch changed GPU lookups; unchanged requests reuse their cached cursor."""
    with DRAFT_CHANGED:
        pending = []
        for buffer, output in zip(buffers, outputs):
            if buffer.error is not None:
                raise buffer.error
            state = buffer._state(output)
            if state == buffer._lookup_state:
                continue
            if buffer.lookup:
                pending.append((buffer, output))
            else:
                buffer.cursor = suffix_lookup(output, buffer.tokens, buffer.last_evaluated,
                                              buffer._text, buffer.lookup_window_width,
                                              min_position=buffer.last_accepted)
                buffer._lookup_state = state
        if pending:
            selected, targets = zip(*pending)
            positions = selected[0].lookup(
                targets, [buffer.tokens for buffer in selected],
                [buffer.last_evaluated for buffer in selected], [buffer.last_accepted for buffer in selected],
            )
            for (buffer, output), position in zip(pending, positions):
                buffer.cursor, buffer._lookup_state = position, buffer._state(output)
        return [buffer.tokens[buffer.cursor:buffer.cursor + max(0, limit)] if buffer.cursor is not None else []
                for buffer, limit in zip(buffers, limits)]
