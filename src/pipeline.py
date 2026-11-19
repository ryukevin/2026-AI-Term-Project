"""
사투리 → 표준어 → 외국어 한 번에 실행 (Gradio 데모와 시연에서 사용)

  from src.pipeline import Pipeline
  pipe = Pipeline()                                   # 정규화 모델 + 번역 모델 로드
  r = pipe.run("니 밥 뭇나", tgt="zh", compare=True)
  r["standard"], r["translation"], r["direct_translation"]

명령줄에서 바로 확인:
  python -m src.pipeline "니 밥 뭇나" "어데 가노" --tgt zh --compare

- 음성(Whisper)은 앞단에 따로 붙인다. 이 모듈은 텍스트부터 시작한다.
- compare=True면 사투리를 그대로 번역한 결과(A 조건)도 같이 돌려준다. 정규화 효과를 화면에서 보여줄 때 쓴다.
- 번역 모델 기본값은 MADLAD (11/19 비교 결과). 평가(evaluate.py)와 같은 모델·디코딩 설정이어야 한다.
- VRAM: KoBART 정규화 + MADLAD 3B(bf16) ≈ 6.5GB. 8GB GPU에서 Whisper까지 같이 올리면 부족할 수 있다.
"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.normalize import Normalizer  # noqa: E402
from src.translate import LANGS, Translator  # noqa: E402

DEFAULT_NORM = "checkpoints/kobart_norm/best"
DEFAULT_MT = "madlad"


class Pipeline:
    def __init__(self, norm_ckpt: str = DEFAULT_NORM, mt_model: str = DEFAULT_MT,
                 mt_dtype: str = "auto", device: str | None = None, load_mt: bool = True):
        ckpt = Path(norm_ckpt)
        if not ckpt.is_absolute():
            ckpt = ROOT / ckpt
        self.normalizer = Normalizer(str(ckpt), device=device)
        self.translator = Translator(mt_model, dtype=mt_dtype, device=device) if load_mt else None

    @staticmethod
    def languages() -> list[str]:
        return list(LANGS)

    def run(self, text: str | list[str], tgt: str = "zh", compare: bool = False) -> dict | list[dict]:
        """문장 하나면 dict, 리스트면 dict 리스트.

        dict 키: dialect, standard, translation, (compare=True면) direct_translation,
                 sec_norm, sec_mt (문장당 평균 초)
        """
        single = isinstance(text, str)
        sents = [text] if single else list(text)
        sents = [str(s or "").strip() for s in sents]
        n = max(len(sents), 1)

        std, sec_norm = self.normalizer.normalize(sents)
        rows = [{"dialect": d, "standard": s, "sec_norm": round(sec_norm / n, 3)} for d, s in zip(sents, std)]

        if self.translator is not None:
            if tgt not in LANGS:
                raise ValueError(f"지원하지 않는 언어 {tgt!r}. 가능한 값: {list(LANGS)}")
            idx = [i for i, s in enumerate(std) if s]
            outs, sec_mt = self.translator.translate([std[i] for i in idx], tgt) if idx else ([], 0.0)
            trans = [""] * len(sents)
            for i, o in zip(idx, outs):
                trans[i] = o
            for r, t in zip(rows, trans):
                r["translation"] = t
                r["sec_mt"] = round(sec_mt / n, 3)
            if compare:
                idx = [i for i, s in enumerate(sents) if s]
                direct, _ = self.translator.translate([sents[i] for i in idx], tgt) if idx else ([], 0.0)
                dt = [""] * len(sents)
                for i, o in zip(idx, direct):
                    dt[i] = o
                for r, t in zip(rows, dt):
                    r["direct_translation"] = t
        return rows[0] if single else rows


def _main():
    import argparse

    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    ap = argparse.ArgumentParser()
    ap.add_argument("text", nargs="+")
    ap.add_argument("--tgt", default="zh")
    ap.add_argument("--norm_ckpt", default=DEFAULT_NORM)
    ap.add_argument("--mt", default=DEFAULT_MT)
    ap.add_argument("--compare", action="store_true", help="사투리 직접 번역(A)도 같이 출력")
    ap.add_argument("--no_mt", action="store_true", help="번역 모델 없이 정규화만")
    args = ap.parse_args()

    pipe = Pipeline(args.norm_ckpt, args.mt, load_mt=not args.no_mt)
    for r in pipe.run(args.text, args.tgt, args.compare):
        print(f"사투리   {r['dialect']}")
        print(f"표준어   {r['standard']}   ({r['sec_norm']}초)")
        if "translation" in r:
            print(f"번역     {r['translation']}   ({r['sec_mt']}초)")
        if "direct_translation" in r:
            print(f"직접번역 {r['direct_translation']}   (정규화 없이)")
        print()


if __name__ == "__main__":
    _main()
