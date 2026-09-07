from lib.classes.sentence_batcher import BlockJob, SentenceBatcher

# chapter sentence counts taken from a real conversion of Outliers-story-of-success.epub
OUTLIERS_HISTOGRAM = (
    [1] * 34
    + [2] * 2
    + [15, 24, 30, 31, 33, 35, 36, 37, 40, 41, 51, 65, 68, 68, 70, 71, 81, 102]
    + [109] * 3
    + [137]
)


def make_job(block_index, sentence_count, voice=None, pending=None, start_sentence=0):
    sentences = [f"block {block_index} sentence {i}." for i in range(sentence_count)]
    if pending is None:
        pending = list(range(sentence_count))
    return BlockJob(
        block_index,
        f"block-{block_index}",
        f"/tmp/sentences/block-{block_index}",
        block_index + 1,
        voice,
        pending,
        sentences,
        sentence_count,
        "flac",
        0,
        max(sentence_count - 1, 0),
        start_sentence,
        False,
    )


def run_pack(batcher, pack, batch_size):
    # stand-in for the engine: split the pack into batch_size groups like
    # Breeze.plan_chunks does, and report each group complete
    groups = [pack[i : i + batch_size] for i in range(0, len(pack), batch_size)]
    for group in groups:
        batcher.mark_done(group)
    return groups


def drain(batcher):
    finished = batcher.take_completed()
    for job in finished:
        batcher.confirm_combined(job)
    return finished


def test_batch_size_one_matches_per_block_dispatch():
    # every engine except breeze reports batch_size 1 and must be unaffected:
    # one pack per block, in block order, each block completing on its own
    batcher = SentenceBatcher(batch_size=1)
    order = []
    for index, count in enumerate([3, 1, 5]):
        job = make_job(index, count)
        packs = batcher.admit(job)
        assert len(packs) == 1
        assert [item.sentence_index for item in packs[0]] == list(range(count))
        run_pack(batcher, packs[0], 1)
        order.extend(job.block_index for job in drain(batcher))
    assert batcher.flush() == []
    assert order == [0, 1, 2]


def test_occupancy_on_measured_histogram():
    batch_size = 64
    batcher = SentenceBatcher(batch_size=batch_size, window_multiplier=4)
    dispatched = []
    groups = []
    for index, count in enumerate(OUTLIERS_HISTOGRAM):
        for pack in batcher.admit(make_job(index, count)):
            dispatched.extend(pack)
            groups.extend(run_pack(batcher, pack, batch_size))
        drain(batcher)
    for pack in batcher.flush():
        dispatched.extend(pack)
        groups.extend(run_pack(batcher, pack, batch_size))
    drain(batcher)
    total = sum(OUTLIERS_HISTOGRAM)
    # every sentence dispatched exactly once
    assert len(dispatched) == total
    assert (
        len({(item.job.block_index, item.sentence_index) for item in dispatched})
        == total
    )
    # dispatching this book per block needs three times the batches, nearly all
    # of them partial; batching across blocks leaves only the final one short
    assert len(groups) == 22
    assert sum(1 for group in groups if len(group) == batch_size) == 21


def test_voice_change_flushes_and_never_mixes():
    batcher = SentenceBatcher(batch_size=8, window_multiplier=4)
    packs = []
    packs.extend(batcher.admit(make_job(0, 4, voice="alice")))
    packs.extend(batcher.admit(make_job(1, 4, voice="alice")))
    # a different voice cannot ride along in the resident batch
    packs.extend(batcher.admit(make_job(2, 4, voice="bob")))
    packs.extend(batcher.flush())
    assert len(packs) == 2
    for pack in packs:
        assert len({item.job.voice for item in pack}) == 1
    assert [item.job.voice for item in packs[0]] == ["alice"] * 8
    assert [item.job.voice for item in packs[1]] == ["bob"] * 4


def test_watermark_holds_at_earliest_unfinished_block():
    batcher = SentenceBatcher(batch_size=4, window_multiplier=4)
    first = make_job(0, 8)
    second = make_job(1, 8)
    packs = batcher.admit(first) + batcher.admit(second)
    pack = packs[0]
    assert len(pack) == 16
    # finish the later block first, which global sorting makes likely
    batcher.mark_done([item for item in pack if item.job is second])
    assert batcher.watermark() == (0, 0)
    # a hole in the earlier block must not let the watermark past it
    batcher.mark_done(
        [item for item in pack if item.job is first and item.sentence_index != 3]
    )
    assert batcher.watermark() == (0, 2)
    batcher.mark_done(
        [item for item in pack if item.job is first and item.sentence_index == 3]
    )
    assert batcher.watermark() == (0, 7)


def test_watermark_needs_the_combine_not_just_the_conversion():
    batcher = SentenceBatcher(batch_size=4, window_multiplier=4)
    first = make_job(0, 4)
    second = make_job(1, 4)
    # 8 items do not fill a 16-item window, so the tail needs a flush
    packs = batcher.admit(first) + batcher.admit(second) + batcher.flush()
    for pack in packs:
        run_pack(batcher, pack, 4)
    assert [job.block_index for job in batcher.take_completed()] == [0, 1]
    # converted but uncombined: advancing here would delete the block on restart
    assert batcher.watermark() == (0, 3)
    batcher.confirm_combined(first)
    assert batcher.watermark() == (1, 3)
    batcher.confirm_combined(second)
    assert batcher.watermark() == (1, 3)


def test_resume_index_reports_start_before_anything_completes():
    batcher = SentenceBatcher(batch_size=4, window_multiplier=4)
    job = make_job(7, 10, pending=[5, 6, 7, 8, 9], start_sentence=5)
    batcher.admit(job)
    assert batcher.watermark() == (7, 5)


def test_empty_pending_block_completes_without_stalling_the_anchor():
    batcher = SentenceBatcher(batch_size=4, window_multiplier=4)
    empty = make_job(0, 4, pending=[])
    assert batcher.admit(empty) == []
    assert [job.block_index for job in batcher.take_completed()] == [0]
    batcher.confirm_combined(empty)
    later = make_job(1, 4)
    for pack in batcher.admit(later) + batcher.flush():
        run_pack(batcher, pack, 4)
    assert [job.block_index for job in batcher.take_completed()] == [1]
    batcher.confirm_combined(later)
    assert batcher.watermark() == (1, 3)


def test_flush_drains_every_admitted_job_exactly_once():
    batcher = SentenceBatcher(batch_size=16, window_multiplier=4)
    counts = [3, 40, 1, 1, 17, 2]
    dispatched = []
    completed = []
    for index, count in enumerate(counts):
        for pack in batcher.admit(make_job(index, count)):
            dispatched.extend(pack)
            run_pack(batcher, pack, 16)
        completed.extend(job.block_index for job in drain(batcher))
    for pack in batcher.flush():
        dispatched.extend(pack)
        run_pack(batcher, pack, 16)
    completed.extend(job.block_index for job in drain(batcher))
    assert len(dispatched) == sum(counts)
    assert sorted(completed) == list(range(len(counts)))
    assert len(completed) == len(set(completed))
    assert batcher.flush() == []
