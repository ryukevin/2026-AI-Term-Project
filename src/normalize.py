"""
사투리 → 표준어 정규화 모듈 (우리가 직접 파인튜닝하는 모델)

학습: scripts/train_normalizer.py  →  checkpoints/<이름>/best/
추론: Normalizer("checkpoints/kobart_norm/best").normalize(["사투리 문장", ...])

학습과 추론이 입력을 똑같이 만들도록 토큰화 함수(encode_batch)를 여기 한 곳에 둔다.
체크포인트 폴더의 norm_config.json에 접두어·최대 길이를 저장하고, 추론 때 그대로 읽는다.

KoBART(gogamza/kobart-base-v2) 주의점
- 토크나이저가 문장 끝 </s>를 자동으로 붙이지 않는다 → 직접 붙인다.
- token_type_ids를 돌려주는데 BART 모델은 이 인자를 받지 않는다 → 버린다.
"""
from __future__ import annotations

import json
import time
from pathlib import Path

# 평가용 고정 디코딩 설정 (B 조건 결과를 낸 뒤에는 바꾸지 않는다. 바꾸면 B를 다시 돌린다)
NORM_DECODE = {"num_beams": 4, "max_new_tokens": 128, "do_sample": False}

CONFIG_NAME = "norm_config.json"

# T5 계열은 과제 접두어를 붙이는 게 관례라 계획서 형식을 쓰고, BART 계열은 붙이지 않는다.
T5_PREFIX = "방언을 표준어로 바꾸기: "


def default_prefix(model_name: str) -> str:
    return T5_PREFIX if "t5" in model_name.lower() else ""


def encode_batch(tok, texts: list[str], max_len: int, prefix: str = "") -> list[list[int]]:
    """문장 목록 → 토큰 ID 목록. 끝에 </s>가 없으면 붙이고, 길면 자른다(</s>는 남김)."""
    eos = tok.eos_token_id
    out = []
    for t in texts:
        ids = tok(prefix + str(t), add_special_tokens=True, return_token_type_ids=False)["input_ids"]
        if eos is not None:
            if ids and ids[-1] == eos:
                ids = ids[:-1]
            ids = ids[: max_len - 1] + [eos]
        else:
            ids = ids[:max_len]
        out.append(ids)
    return out


def pad_batch(seqs: list[list[int]], pad_id: int, label: bool = False):
    """동적 패딩. label=True면 패딩 자리를 -100으로 채워 손실 계산에서 뺀다."""
    import torch

    n = max(len(s) for s in seqs)
    fill = -100 if label else pad_id
    ids = torch.full((len(seqs), n), fill, dtype=torch.long)
    mask = torch.zeros((len(seqs), n), dtype=torch.long)
    for i, s in enumerate(seqs):
        ids[i, : len(s)] = torch.tensor(s, dtype=torch.long)
        mask[i, : len(s)] = 1
    return ids, mask


def generate(model, tok, texts: list[str], max_len: int, prefix: str = "",
             batch_size: int = 32, decode: dict | None = None) -> list[str]:
    """학습 중 검증과 추론에서 같이 쓰는 생성 함수"""
    import torch

    decode = {**NORM_DECODE, **(decode or {})}
    pad = tok.pad_token_id if tok.pad_token_id is not None else tok.eos_token_id
    extra = {"pad_token_id": pad}
    if tok.eos_token_id is not None:
        extra["eos_token_id"] = tok.eos_token_id
    if getattr(model.config, "decoder_start_token_id", None) is None:
        extra["decoder_start_token_id"] = tok.eos_token_id   # BART 관례
    outs = []
    dev = next(model.parameters()).device
    for i in range(0, len(texts), batch_size):
        batch = [str(t) for t in texts[i:i + batch_size]]
        ids, mask = pad_batch(encode_batch(tok, batch, max_len, prefix), pad)
        with torch.inference_mode():
            gen = model.generate(input_ids=ids.to(dev), attention_mask=mask.to(dev), **extra, **decode)
        outs.extend(s.strip() for s in tok.batch_decode(gen, skip_special_tokens=True))
    return outs


class Normalizer:
    def __init__(self, ckpt: str = "checkpoints/kobart_norm/best", dtype: str = "auto",
                 device: str | None = None):
        import torch
        from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

        path = Path(ckpt)
        if not path.exists():
            raise FileNotFoundError(f"정규화 모델 체크포인트가 없음: {path}  (scripts/train_normalizer.py 로 먼저 학습)")
        cfg_file = path / CONFIG_NAME
        self.cfg = json.loads(cfg_file.read_text(encoding="utf-8")) if cfg_file.exists() else {}
        self.prefix = self.cfg.get("prefix", "")
        self.max_len = int(self.cfg.get("max_len", 128))

        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        if dtype == "auto":
            # T5는 fp16에서 값이 넘칠 수 있어서 bf16, BART는 fp16. CPU는 fp32.
            if self.device != "cuda":
                dtype = "fp32"
            elif torch.cuda.is_bf16_supported():
                dtype = "bf16"
            else:
                dtype = "fp16"
        self.dtype = dtype

        self.tok = AutoTokenizer.from_pretrained(str(path))
        self.model = AutoModelForSeq2SeqLM.from_pretrained(str(path))
        self.model.to(self.device, dtype={"fp16": torch.float16, "bf16": torch.bfloat16,
                                          "fp32": torch.float32}[dtype])
        self.model.eval()
        self.name = str(path)

    def normalize(self, sents: list[str], batch_size: int = 32,
                  decode: dict | None = None) -> tuple[list[str], float]:
        """표준어 결과 리스트와 총 소요 시간(초)을 돌려준다. 빈 문장은 빈 문자열 그대로."""
        t0 = time.time()
        sents = [str(s or "").strip() for s in sents]
        idx = [i for i, s in enumerate(sents) if s]
        preds = generate(self.model, self.tok, [sents[i] for i in idx], self.max_len,
                         self.prefix, batch_size, decode)
        out = [""] * len(sents)
        for i, p in zip(idx, preds):
            out[i] = p
        return out, time.time() - t0
