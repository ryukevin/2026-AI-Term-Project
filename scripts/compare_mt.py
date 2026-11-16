"""
번역 모델 후보 비교: NLLB 1.3B vs MADLAD 3B (표준어 → 중국어 간체)

같은 표준어 문장 20~30개를 두 모델로 번역해 chrF·속도·VRAM을 비교하고,
중국어 원어민이 눈으로 고를 수 있게 나란히 놓은 CSV를 만든다.

  # 1) 문장 고르기 + 번역 (정답 zh가 아직 없으면 chrF 없이 비교표만 생성)
  python scripts/compare_mt.py --from_pairs data/pairs/test.csv --n 25

  # 2) 원어민이 zh 정답을 채운 파일로 다시 채점만
  python scripts/compare_mt.py --sents data/eval/mt_pilot.csv

출력
  data/eval/mt_pilot.csv            고른 문장 (id, standard, zh) — zh 칸을 원어민이 채운다
  results/mt_compare.csv            문장별 두 모델 번역 나란히 + 원어민 선택 칸(better)
  results/mt_compare_summary.csv    모델별 chrF/BLEU, 문장당 초, 최대 VRAM
"""
import argparse
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.translate import DECODE, MODELS, Translator  # noqa: E402

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")


def pick_sentences(pairs_csv: str, n: int, seed: int) -> pd.DataFrame:
    df = pd.read_csv(pairs_csv)
    # 너무 짧거나 긴 문장 제외, 데이터셋 골고루
    df = df[(df.n_eojeol >= 5) & (df.n_eojeol <= 20)]
    per = max(1, n // df.dataset.nunique())
    pick = df.groupby("dataset", group_keys=False).apply(lambda g: g.sample(min(per, len(g)), random_state=seed))
    pick = pick.head(n)
    return pd.DataFrame({"id": pick.id, "standard": pick.standard, "zh": ""})


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sents", default="data/eval/mt_pilot.csv", help="standard(필수), zh(선택) 열이 있는 CSV")
    ap.add_argument("--from_pairs", help="이 CSV에서 문장을 새로 골라 --sents 파일로 저장")
    ap.add_argument("--n", type=int, default=25)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--models", nargs="+", default=["nllb", "madlad"], help=f"후보: {list(MODELS)} 또는 HF 이름")
    ap.add_argument("--dtype", default="auto", help="auto/fp16/bf16/fp32/int8 (MADLAD 3B는 8GB에서 fp16 또는 int8)")
    ap.add_argument("--batch_size", type=int, default=4)
    ap.add_argument("--out_dir", default="results")
    args = ap.parse_args()

    sp = Path(args.sents)
    if args.from_pairs:
        sp.parent.mkdir(parents=True, exist_ok=True)
        if sp.exists():
            print(f"{sp} 가 이미 있어서 덮어쓰지 않음 (원어민 정답이 들어있을 수 있음). 지우고 다시 실행할 것.")
        else:
            pick_sentences(args.from_pairs, args.n, args.seed).to_csv(sp, index=False, encoding="utf-8-sig")
            print(f"문장 {args.n}개 → {sp} (zh 칸은 원어민이 채움)")
    sents = pd.read_csv(sp).fillna("")
    refs = sents["zh"].astype(str).tolist() if "zh" in sents.columns else []
    has_ref = bool(refs) and all(r.strip() for r in refs)

    import torch
    import sacrebleu

    out = sents[["id", "standard"]].copy() if "id" in sents.columns else sents[["standard"]].copy()
    if has_ref:
        out["zh_ref"] = refs
    rows = []
    for m in args.models:
        print(f"\n=== {m} 로딩 ===", flush=True)
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats()
        tr = Translator(m, dtype=args.dtype)
        tr.translate(sents.standard[:1].tolist(), "zh")      # 워밍업 (속도 측정에서 제외)
        hyps, sec = tr.translate(sents.standard.tolist(), "zh", batch_size=args.batch_size)
        vram = torch.cuda.max_memory_allocated() / 2**30 if torch.cuda.is_available() else float("nan")
        out[m] = hyps
        r = {"model": tr.name, "dtype": tr.dtype, "device": tr.device, "n": len(hyps),
             "sec_per_sent": round(sec / len(hyps), 3), "peak_vram_gb": round(vram, 2),
             "decode": str(DECODE)}
        if has_ref:
            r["chrF"] = round(sacrebleu.corpus_chrf(hyps, [refs]).score, 2)
            r["BLEU_zh"] = round(sacrebleu.corpus_bleu(hyps, [refs], tokenize="zh").score, 2)
        rows.append(r)
        print(r, flush=True)
        for s, h in list(zip(sents.standard, hyps))[:3]:
            print(f"  {s}\n   → {h}")
        del tr
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    out["better"] = ""   # 원어민이 nllb / madlad / same 중 하나를 적는다
    out["memo"] = ""
    od = Path(args.out_dir)
    od.mkdir(exist_ok=True)
    out.to_csv(od / "mt_compare.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame(rows).to_csv(od / "mt_compare_summary.csv", index=False, encoding="utf-8-sig")
    print(f"\n저장: {od / 'mt_compare.csv'}, {od / 'mt_compare_summary.csv'}")
    if not has_ref:
        print("zh 정답이 비어 있어서 chrF는 계산 안 함 → 원어민이 채운 뒤 --sents 로 다시 실행")


if __name__ == "__main__":
    main()
