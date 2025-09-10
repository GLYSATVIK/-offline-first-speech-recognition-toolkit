# Usage guide

## Installation

```bash
pip install chunkasr
# optional extras
pip install "chunkasr[torch]"   # PyTorch backend
pip install "chunkasr[mic]"     # live microphone
```

## The Recognizer

`Recognizer` is the high-level entry point. The `demo()` constructor wires up a
seeded reference model so you can run the whole pipeline with no training:

```python
import chunkasr

rec = chunkasr.Recognizer.demo(seed=1234)
samples, sr = chunkasr.read_wav("utterance.wav")
result = rec.transcribe(samples, sample_rate=sr)

print(result.text)          # decoded string
print(result.rtf)           # real-time factor
for w in result.words:      # word timings
    print(w.word, w.start, w.end)
```

`transcribe` accepts any sample rate and resamples internally to the frontend's
configured rate.

## Choosing a decoder

```python
from chunkasr.config import DecodeConfig

rec = chunkasr.Recognizer.demo(decode=DecodeConfig(method="ctc_beam", beam_size=16))
```

Available methods: `ctc_greedy`, `ctc_beam`, `rnnt_greedy`.

## Language-model fusion

```python
from chunkasr.decode.lm import NGramLM

lm = NGramLM.train_chars(open("corpus.txt").read().splitlines(), order=3)
rec = chunkasr.Recognizer.demo().with_lm(lm, weight=0.4)
```

`with_lm` switches the recogniser to prefix beam search and adds the LM as a
shallow-fusion term.

## Streaming

```python
rec = chunkasr.Recognizer.demo().enable_streaming(chunk_size=16, right_context=4)
result = rec.transcribe(samples, sample_rate=sr)
```

## Feature extraction on its own

```python
from chunkasr.config import FeatureConfig
from chunkasr.frontend import FeatureExtractor

extractor = FeatureExtractor(FeatureConfig(num_mel_bins=80), delta_order=0)
features = extractor(samples)   # (n_frames, 80)
```

## Voice activity detection

```python
from chunkasr.audio import detect_speech

for seg in detect_speech(samples, sample_rate=16000):
    print(seg.start, seg.end)
```

## Scoring

```python
from chunkasr import wer, cer
from chunkasr.metrics import corpus_wer

print(wer("the quick brown fox", "the quick brown fax").as_percent())
print(cer("hello", "hallo"))
print(corpus_wer(references, hypotheses).wer)
```

## Command line

```bash
chunkasr transcribe a.wav b.wav --method ctc_beam --timestamps
chunkasr score --ref refs.txt --hyp hyps.txt
chunkasr manifest validate data/train.jsonl
chunkasr manifest stats data/train.jsonl
chunkasr mic --seconds 5           # requires the [mic] extra
chunkasr info
```

## Checkpoints and configs

```python
from chunkasr.checkpoint import save_checkpoint, load_checkpoint
from chunkasr.config import ExperimentConfig, save_config, load_config

save_checkpoint("model.npz", rec.model)
ckpt = load_checkpoint("model.npz")

save_config(ExperimentConfig(name="exp1"), "exp1.yaml")
cfg = load_config("exp1.yaml")
```
