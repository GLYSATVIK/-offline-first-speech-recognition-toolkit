from __future__ import annotations

import numpy as np

from chunkasr.audio.io import write_wav
from chunkasr.cli.main import main


def test_info() -> None:
    assert main(["info"]) == 0


def test_transcribe_file(tmp_path, capsys) -> None:
    wav = tmp_path / "u.wav"
    write_wav(wav, np.zeros(8000, dtype=np.float32), 16000)
    assert main(["transcribe", str(wav)]) == 0
    assert str(wav) in capsys.readouterr().out


def test_score(tmp_path, capsys) -> None:
    ref = tmp_path / "ref.txt"
    hyp = tmp_path / "hyp.txt"
    ref.write_text("the cat sat\nhello world\n", encoding="utf-8")
    hyp.write_text("the bat sat\nhello world\n", encoding="utf-8")
    assert main(["score", "--ref", str(ref), "--hyp", str(hyp)]) == 0
    assert "WER" in capsys.readouterr().out


def test_score_length_mismatch(tmp_path) -> None:
    ref = tmp_path / "ref.txt"
    hyp = tmp_path / "hyp.txt"
    ref.write_text("a\nb\n", encoding="utf-8")
    hyp.write_text("a\n", encoding="utf-8")
    assert main(["score", "--ref", str(ref), "--hyp", str(hyp)]) == 2


def test_manifest_stats(tmp_path, capsys) -> None:
    manifest = tmp_path / "m.jsonl"
    manifest.write_text(
        '{"audio_filepath": "a.wav", "duration": 1.0, "text": "hi"}\n',
        encoding="utf-8",
    )
    assert main(["manifest", "stats", str(manifest)]) == 0
    assert "num_utterances" in capsys.readouterr().out
