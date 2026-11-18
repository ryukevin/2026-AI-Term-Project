"""
다국어 번역 모듈 (학습 안 함, 사전학습 모델 그대로 사용)

- NLLB (facebook/nllb-200-*) : 소스 kor_Hang, 타깃은 forced_bos_token 으로 지정
- MADLAD (google/madlad400-*-mt) : 입력 앞에 <2zh> 같은 태그

A/B/C 조건 비교에서는 **같은 체크포인트와 같은 디코딩 설정**을 써야 하므로
DECODE 값을 바꾸지 말 것. 바꿨다면 세 조건을 모두 다시 돌린다.
"""
from __future__ import annotations

import time

# 평가용 고정 디코딩 설정 (번역 모델 확정 후 절대 바꾸지 않는다)
DECODE = {"num_beams": 4, "max_new_tokens": 256, "do_sample": False}

# 데모에서 고를 수 있는 언어. 평가는 zh(간체)만 한다.
LANGS = {
    #       NLLB 코드    MADLAD 태그
    "zh": ("zho_Hans", "<2zh>"),
    "en": ("eng_Latn", "<2en>"),
    "ja": ("jpn_Jpan", "<2ja>"),
    "vi": ("vie_Latn", "<2vi>"),
    "th": ("tha_Thai", "<2th>"),
    "ru": ("rus_Cyrl", "<2ru>"),
    "es": ("spa_Latn", "<2es>"),
}

MODELS = {
    "nllb": "facebook/nllb-200-distilled-1.3B",
    "nllb600m": "facebook/nllb-200-distilled-600M",   # 메모리 부족한 PC 테스트용
    "madlad": "google/madlad400-3b-mt",
}


class Translator:
    def __init__(self, model: str = "nllb", dtype: str = "auto", device: str | None = None):
        import torch
        from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

        self.name = MODELS.get(model, model)
        self.kind = "madlad" if "madlad" in self.name.lower() else "nllb"
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        if dtype == "auto":
            if self.device != "cuda":
                dtype = "fp32"
            elif self.kind == "madlad" and torch.cuda.is_bf16_supported():
                # MADLAD(T5 계열)는 fp16에서 값이 넘쳐 빈 번역이 나올 수 있어서 bf16 사용 (RTX 30xx 지원)
                dtype = "bf16"
            else:
                dtype = "fp16"

        kw = {}
        if dtype == "int8":
            from transformers import BitsAndBytesConfig
            kw["quantization_config"] = BitsAndBytesConfig(load_in_8bit=True)
            kw["device_map"] = "auto"
        else:
            kw["torch_dtype"] = {"fp16": torch.float16, "bf16": torch.bfloat16, "fp32": torch.float32}[dtype]
        self.dtype = dtype

        tok_kw = {"src_lang": "kor_Hang"} if self.kind == "nllb" else {}
        self.tok = AutoTokenizer.from_pretrained(self.name, **tok_kw)
        self.model = AutoModelForSeq2SeqLM.from_pretrained(self.name, **kw)
        if dtype != "int8":
            self.model.to(self.device)
        self.model.eval()

    def translate(self, sents: list[str], tgt: str = "zh", batch_size: int = 8,
                  decode: dict | None = None) -> tuple[list[str], float]:
        """번역 결과 리스트와 총 소요 시간(초)을 돌려준다."""
        import torch

        decode = {**DECODE, **(decode or {})}
        nllb_code, madlad_tag = LANGS[tgt]
        outs, t0 = [], time.time()
        for i in range(0, len(sents), batch_size):
            batch = [str(s) for s in sents[i:i + batch_size]]
            if self.kind == "madlad":
                batch = [f"{madlad_tag} {s}" for s in batch]
                extra = {}
            else:
                extra = {"forced_bos_token_id": self.tok.convert_tokens_to_ids(nllb_code)}
            enc = self.tok(batch, return_tensors="pt", padding=True, truncation=True,
                           max_length=256).to(self.model.device)
            with torch.inference_mode():
                gen = self.model.generate(**enc, **extra, **decode)
            outs.extend(self.tok.batch_decode(gen, skip_special_tokens=True))
        return [o.strip() for o in outs], time.time() - t0
