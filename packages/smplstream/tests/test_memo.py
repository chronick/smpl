"""Memo key + parameter canonicalization (spec → *Memoization*, NORMATIVE)."""

from __future__ import annotations

from smplstream import memo


def test_param_spellings_collapse():
    # 6.0 ≡ 6, key order irrelevant → one canonical string → one key.
    a = memo.memo_key("eq", "eq@1", ["blake3:" + "a" * 64], {"gain": 6.0, "freq": 1000})
    b = memo.memo_key("eq", "eq@1", ["blake3:" + "a" * 64], {"freq": 1000, "gain": 6})
    assert a == b


def test_input_order_normalized():
    ins = ["blake3:" + "b" * 64, "blake3:" + "a" * 64]
    a = memo.memo_key("mix", "mix@1", ins, {})
    b = memo.memo_key("mix", "mix@1", list(reversed(ins)), {})
    assert a == b


def test_op_version_changes_key():
    a = memo.memo_key("demucs", "audio-separator@0.28+htdemucs:blake3:aa", ["blake3:" + "a" * 64], {})
    b = memo.memo_key("demucs", "audio-separator@0.29+htdemucs:blake3:bb", ["blake3:" + "a" * 64], {})
    assert a != b


def test_env_fingerprint_changes_key():
    base = dict(op="sox-reverb", op_version="reverb@1", input_hashes=["blake3:" + "a" * 64], params={})
    assert memo.memo_key(**base, env_fingerprint="ffmpeg-8") != memo.memo_key(**base, env_fingerprint="ffmpeg-7")


def test_set_params_sorted_sequence_preserved():
    # A set-valued param is order-insensitive; a sequence param is order-sensitive.
    s1 = memo.canonicalize_params({"tags": ["b", "a"]}, set_keys=["tags"])
    s2 = memo.canonicalize_params({"tags": ["a", "b"]}, set_keys=["tags"])
    assert s1 == s2
    q1 = memo.canonicalize_params({"steps": ["b", "a"]})
    q2 = memo.canonicalize_params({"steps": ["a", "b"]})
    assert q1 != q2


def test_canonical_number_forms():
    assert memo.canonicalize_params({"x": 6.0}) == memo.canonicalize_params({"x": 6})
    assert memo.canonicalize_params({"x": 6.5}) != memo.canonicalize_params({"x": 6})
