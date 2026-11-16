"""
A/B/C(/D) 조건 비교 평가

평가 세트 CSV (data/eval/eval_set.csv) 열:
  id, dialect, standard, zh          ← 필수 (zh = 중국어 정답, 원어민 작성)
  pred_standard                      ← 선택 (B 조건: 우리 정규화 모델 출력. --pred 파일로 줘도 됨)
  rule_standard                      ← 선택 (D 조건: 사전 단순 치환)

조건
  A  사투리 원문 그대로         (기준선)
  B  모델이 변환한 표준어       (제안 방법)
  C  사람이 쓴 표준어 정답      (상한선)
  D  사전 치환                  (있으면)

사용
  # 번역까지 한 번에 (번역 결과는 results/translations_<tag>.csv 에 캐시)
  python scripts/evaluate.py --eval data/eval/eval_set.csv --translate --model nllb --tag nllb
  # B 조건 입력을 따로 줄 때 (id, pred_standard 열)
  python scripts/evaluate.py --eval data/eval/eval_set.csv --pred results/kobart_pred.csv --translate
  # 원어민 블라인드 평가지 만들기
  python scripts/evaluate.py --eval data/eval/eval_set.csv --tag nllb --human_sheet

출력
  results/scores_<tag>.csv          조건별 BLEU(zh)·chrF + 95% 부트스트랩 구간
  results/norm_scores_<tag>.csv     변환 단계 점수 (표준어 정답과 비교한 BLEU·CER·완전일치)
  results/eval_detail_<tag>.csv     문장별 번역과 chrF
  results/human_eval_blind_<tag>.csv / human_eval_key_<tag>.csv
"""
import argparse
import random
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.aihub import norm_key  # noqa: E402

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

COND_INPUT = {"A": "dialect", "B": "pred_standard", "C": "standard", "D": "rule_standard"}
COND_NAME = {"A": "사투리 원문", "B": "모델 변환 표준어", "C": "정답 표준어", "D": "사전 치환"}


def mt_scores(hyps, refs):
    import sacrebleu
    return (sacrebleu.corpus_bleu(hyps, [refs], tokenize="zh").score,
            sacrebleu.corpus_chrf(hyps, [refs]).score)


def bootstrap(hyps, refs, n=1000, seed=0):
    """문장 재표집으로 BLEU·chrF 95% 구간"""
    rng = random.Random(seed)
    idx = list(range(len(refs)))
    bs, cs = [], []
    for _ in range(n):
        s = [rng.choice(idx) for _ in idx]
        b, c = mt_scores([hyps[i] for i in s], [refs[i] for i in s])
        bs.append(b)
        cs.append(c)
    q = lambda v: (sorted(v)[int(0.025 * n)], sorted(v)[int(0.975 * n) - 1])  # noqa: E731
    return q(bs), q(cs)


def paired_diff(h1, h2, refs, n=1000, seed=0):
    """chrF(h2) - chrF(h1) 의 부트스트랩 95% 구간과 h2가 더 높은 비율"""
    import sacrebleu
    rng = random.Random(seed)
    idx = list(range(len(refs)))
    diffs = []
    for _ in range(n):
        s = [rng.choice(idx) for _ in idx]
        r = [[refs[i] for i in s]]
        diffs.append(sacrebleu.corpus_chrf([h2[i] for i in s], r).score
                     - sacrebleu.corpus_chrf([h1[i] for i in s], r).score)
    diffs.sort()
    return diffs[int(0.025 * n)], diffs[int(0.975 * n) - 1], sum(d > 0 for d in diffs) / n


def norm_scores(pred, ref):
    import jiwer
    import sacrebleu
    p = [norm_key(x) for x in pred]
    r = [norm_key(x) for x in ref]
    return {
        "BLEU_word": round(sacrebleu.corpus_bleu(list(pred), [list(ref)]).score, 2),
        "BLEU_char": round(sacrebleu.corpus_bleu(list(pred), [list(ref)], tokenize="char").score, 2),
        "CER": round(jiwer.cer(r, p), 4),
        "exact_match": round(sum(a == b for a, b in zip(p, r)) / len(r), 4),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--eval", default="data/eval/eval_set.csv")
    ap.add_argument("--pred", help="B 조건 입력 CSV (id, pred_standard)")
    ap.add_argument("--translate", action="store_true", help="없는 조건의 번역을 새로 만든다")
    ap.add_argument("--model", default="nllb", help="번역 모델 (확정 후 고정!)")
    ap.add_argument("--dtype", default="auto")
    ap.add_argument("--tgt", default="zh")
    ap.add_argument("--tag", default="nllb")
    ap.add_argument("--n_boot", type=int, default=1000)
    ap.add_argument("--human_sheet", action="store_true")
    ap.add_argument("--out_dir", default="results")
    args = ap.parse_args()

    od = Path(args.out_dir)
    od.mkdir(exist_ok=True)
    ev = pd.read_csv(args.eval).fillna("")
    for col in ("id", "dialect", "standard", "zh"):
        if col not in ev.columns:
            sys.exit(f"평가 세트에 '{col}' 열이 없음")
    ev["id"] = ev["id"].astype(str)
    if args.pred:
        p = pd.read_csv(args.pred).fillna("")
        p["id"] = p["id"].astype(str)
        ev = ev.drop(columns=["pred_standard"], errors="ignore").merge(p[["id", "pred_standard"]], on="id", how="left")
        ev["pred_standard"] = ev["pred_standard"].fillna("")

    conds = [c for c, col in COND_INPUT.items() if col in ev.columns and (ev[col].astype(str).str.strip() != "").all()]
    print(f"평가 문장 {len(ev)}개, 사용 가능한 조건: {conds}")

    # ---------------- 번역 (캐시)
    cache = od / f"translations_{args.tag}.csv"
    tr_df = pd.read_csv(cache, dtype={"id": str}).fillna("") if cache.exists() else pd.DataFrame({"id": ev.id})
    tr_df = ev[["id"]].merge(tr_df, on="id", how="left").fillna("")
    for c in conds:
        col = f"zh_{c}"
        if col in ev.columns and (ev[col] != "").all():         # 평가 세트에 이미 번역이 있음
            tr_df[col] = ev[col].values
        if col in tr_df.columns and (tr_df[col] != "").all():
            continue
        if not args.translate:
            print(f"  {c} 조건 번역 없음 → --translate 로 만들 것 (건너뜀)")
            continue
        if "translator" not in locals():
            from src.translate import DECODE, Translator
            translator = Translator(args.model, dtype=args.dtype)
            print(f"번역 모델 {translator.name} ({translator.dtype}), 디코딩 {DECODE}")
        hyps, sec = translator.translate(ev[COND_INPUT[c]].tolist(), args.tgt)
        tr_df[col] = hyps
        print(f"  {c} 조건 번역 완료 ({sec:.1f}초)")
        tr_df.to_csv(cache, index=False, encoding="utf-8-sig")
    conds = [c for c in conds if f"zh_{c}" in tr_df.columns and (tr_df[f"zh_{c}"] != "").all()]
    if not conds:
        sys.exit("채점할 번역이 없음")

    # ---------------- 번역 점수
    import sacrebleu
    refs = ev["zh"].tolist()
    rows, detail = [], ev[["id", "dialect", "standard", "zh"]].copy()
    for c in conds:
        hyps = tr_df[f"zh_{c}"].tolist()
        b, ch = mt_scores(hyps, refs)
        (bl, bh), (cl, chh) = bootstrap(hyps, refs, args.n_boot)
        rows.append({"condition": c, "name": COND_NAME[c], "n": len(hyps),
                     "BLEU_zh": round(b, 2), "BLEU_95CI": f"{bl:.1f}~{bh:.1f}",
                     "chrF": round(ch, 2), "chrF_95CI": f"{cl:.1f}~{chh:.1f}"})
        detail[f"zh_{c}"] = hyps
        detail[f"chrF_{c}"] = [round(sacrebleu.sentence_chrf(h, [r]).score, 1) for h, r in zip(hyps, refs)]
    scores = pd.DataFrame(rows)

    # B가 A와 C 사이 어디에 있는지 (chrF 기준)
    if {"A", "B", "C"} <= set(conds):
        a, bb, cc = (scores.set_index("condition").chrF[k] for k in "ABC")
        gap = (bb - a) / (cc - a) * 100 if cc != a else float("nan")
        print(f"\nB가 A→C 격차의 {gap:.1f}%를 메움 (chrF 기준)")
    pairs = [(x, y) for x, y in (("A", "B"), ("B", "C"), ("A", "C"), ("A", "D"), ("D", "B")) if x in conds and y in conds]
    diff_rows = []
    for x, y in pairs:
        lo, hi, p = paired_diff(tr_df[f"zh_{x}"].tolist(), tr_df[f"zh_{y}"].tolist(), refs, args.n_boot)
        diff_rows.append({"compare": f"{y}-{x}", "chrF_diff_95CI": f"{lo:+.2f}~{hi:+.2f}", "P(diff>0)": round(p, 3)})

    scores.to_csv(od / f"scores_{args.tag}.csv", index=False, encoding="utf-8-sig")
    if diff_rows:
        pd.DataFrame(diff_rows).to_csv(od / f"scores_diff_{args.tag}.csv", index=False, encoding="utf-8-sig")
    detail.to_csv(od / f"eval_detail_{args.tag}.csv", index=False, encoding="utf-8-sig")
    print("\n=== 번역 점수 (중국어) ===")
    print(scores.to_string(index=False))
    if diff_rows:
        print(pd.DataFrame(diff_rows).to_string(index=False))

    # ---------------- 변환 단계 점수 (번역 점수와 따로 기록)
    nrows = []
    for c in ("A", "B", "D"):
        if c in COND_INPUT and COND_INPUT[c] in ev.columns and (ev[COND_INPUT[c]] != "").all():
            r = norm_scores(ev[COND_INPUT[c]].tolist(), ev["standard"].tolist())
            nrows.append({"condition": c, "name": COND_NAME[c] + (" (그대로 복사 = 기준)" if c == "A" else ""), **r})
    if nrows:
        ns = pd.DataFrame(nrows)
        ns.to_csv(od / f"norm_scores_{args.tag}.csv", index=False, encoding="utf-8-sig")
        print("\n=== 변환 단계 점수 (정답 표준어 대비) ===")
        print(ns.to_string(index=False))
        print("※ A는 아무것도 안 바꾼 복사 기준선. BLEU가 높게 나와도 B가 A보다 높아야 의미가 있다.")

    # ---------------- 원어민 블라인드 평가지
    if args.human_sheet:
        rng = random.Random(2026)
        items = [(r.id, c, r.dialect, r.standard, tr_df.loc[i, f"zh_{c}"]) for i, r in ev.iterrows() for c in conds]
        rng.shuffle(items)
        blind = pd.DataFrame([{"item": f"H{k:04d}", "korean_dialect": d, "korean_meaning": s, "chinese_output": h,
                               "의미정확성(1-5)": "", "자연스러움(1-5)": "", "말투·의도보존(1-5)": "", "memo": ""}
                              for k, (_, _, d, s, h) in enumerate(items, 1)])
        key = pd.DataFrame([{"item": f"H{k:04d}", "id": i, "condition": c}
                            for k, (i, c, _, _, _) in enumerate(items, 1)])
        blind.to_csv(od / f"human_eval_blind_{args.tag}.csv", index=False, encoding="utf-8-sig")
        key.to_csv(od / f"human_eval_key_{args.tag}.csv", index=False, encoding="utf-8-sig")
        print(f"\n원어민 평가지 {len(blind)}행 → human_eval_blind_{args.tag}.csv (정답 키는 평가자에게 주지 말 것)")


if __name__ == "__main__":
    main()
