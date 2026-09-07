from __future__ import annotations

import os

DEFAULT_WINDOW_MULTIPLIER = 4


def resolve_window_multiplier(default: int = DEFAULT_WINDOW_MULTIPLIER) -> int:
    raw = os.environ.get("E2A_TTS_BATCH_WINDOW")
    if not raw:
        return default
    try:
        value = int(raw)
    except ValueError:
        print(f"Ignoring E2A_TTS_BATCH_WINDOW={raw!r}: not an integer, using {default}")
        return default
    if value < 1:
        print(f"Ignoring E2A_TTS_BATCH_WINDOW={value}: must be >= 1, using {default}")
        return default
    return value


class Item:
    def __init__(
        self, job: BlockJob, sentence_index: int, path: str, text: str
    ) -> None:
        self.job = job
        self.sentence_index = sentence_index
        self.path = path
        self.text = text


class BlockJob:
    # one kept block's TTS work, tracked while its sentences are in flight

    def __init__(
        self,
        block_index: int,
        block_id: str,
        block_dir: str,
        ch_num: int,
        voice: str | None,
        pending: list,
        sentences: list,
        block_len: int,
        ext: str,
        sent_start: int,
        sent_end: int,
        start_sentence: int,
        needs_combine: bool,
    ) -> None:
        self.block_index = block_index
        self.block_id = block_id
        self.block_dir = block_dir
        self.ch_num = ch_num
        self.voice = voice
        self.pending = list(pending)
        self.sentences = sentences
        self.block_len = block_len
        self.ext = ext
        self.sent_start = sent_start
        self.sent_end = sent_end
        self.start_sentence = start_sentence
        self.needs_combine = needs_combine
        self.converted = False
        self.combined = False
        self.done = set()
        self.frontier = 0

    def items(self) -> list:
        return [
            Item(
                self,
                j,
                os.path.join(self.block_dir, f"{j}.{self.ext}"),
                self.sentences[j].strip(),
            )
            for j in self.pending
        ]

    def mark(self, sentence_index: int) -> None:
        # chunks finish out of order and one index cannot describe progress with a
        # hole in it, so the frontier stops at the first unfinished sentence.
        self.done.add(sentence_index)
        while (
            self.frontier < len(self.pending)
            and self.pending[self.frontier] in self.done
        ):
            self.frontier += 1

    @property
    def is_converted(self) -> bool:
        return self.frontier >= len(self.pending)

    def resume_index(self) -> int:
        # records the last completed sentence rather than the next one, so a resume
        # always re-does one sentence. upstream's `sentence_resume = j` did the same.
        if self.frontier == 0:
            return self.start_sentence
        return self.pending[self.frontier - 1]


class SentenceBatcher:
    # Fills batches across block boundaries. Dispatching per block leaves most of
    # every batch empty, because each chapter pays for a partial tail batch and a
    # title-stub chapter pays for a whole batch to carry one sentence.

    def __init__(
        self, batch_size: int, window_multiplier: int = DEFAULT_WINDOW_MULTIPLIER
    ) -> None:
        self.batch_size = max(1, int(batch_size))
        self.window_multiplier = max(1, int(window_multiplier))
        # a batch runs until its longest sequence finishes, so sorting only pays off
        # when the sorted population is larger than one batch. buffer several
        # batches' worth, let the engine sort the whole window, then emit full
        # batches. at batch_size 1 there is nothing to fill or sort, so dispatch on
        # admission and keep the one-at-a-time behaviour of every other engine.
        self.window = (
            self.batch_size * self.window_multiplier if self.batch_size > 1 else 1
        )
        self._buffer = []
        self._jobs = []
        self._completed = []
        self._voice = None
        self._last_durable = None

    def admit(self, job: BlockJob) -> list:
        packs = []
        if self._buffer and job.voice != self._voice:
            # a batch cannot mix voices
            packs.extend(self.flush())
        self._voice = job.voice
        self._jobs.append(job)
        if not job.pending:
            # nothing to synthesize, but a changed block still needs recombining
            self._completed.append(job)
            return packs
        self._buffer.extend(job.items())
        packs.extend(self._emit_full_batches())
        return packs

    def _emit_full_batches(self) -> list:
        packs = []
        while len(self._buffer) >= self.window:
            # whole batches only: the remainder stays buffered for the next window
            size = (len(self._buffer) // self.batch_size) * self.batch_size
            packs.append(self._buffer[:size])
            del self._buffer[:size]
        return packs

    def flush(self) -> list:
        # one pack: the engine splits it at batch_size, so the widest possible sort
        if not self._buffer:
            return []
        pack = self._buffer
        self._buffer = []
        return [pack]

    def mark_done(self, items: list) -> None:
        touched = []
        for item in items:
            item.job.mark(item.sentence_index)
            if item.job not in touched:
                touched.append(item.job)
        for job in touched:
            if job.is_converted and job not in self._completed:
                self._completed.append(job)

    def take_completed(self) -> list:
        # ascending block order: packs complete out of order, chapters must not
        ready = sorted(self._completed, key=lambda job: job.block_index)
        self._completed = []
        return ready

    def confirm_combined(self, job: BlockJob) -> None:
        # only a combined block is durable. advancing past a converted-but-uncombined
        # block would lose it: on restart `x < block_resume` with no chapter file
        # resets the block and deletes its whole sentence directory.
        job.combined = True
        while self._jobs and self._jobs[0].combined:
            self._last_durable = self._jobs.pop(0)

    def watermark(self) -> tuple | None:
        # the anchor is the earliest block that is not durable yet
        if self._jobs:
            anchor = self._jobs[0]
            return anchor.block_index, anchor.resume_index()
        if self._last_durable is not None:
            return self._last_durable.block_index, self._last_durable.resume_index()
        return None
