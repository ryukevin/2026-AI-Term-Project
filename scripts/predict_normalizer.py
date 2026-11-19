"""
학습한 정규화 모델로 사투리 → 표준어 변환 (B 조건 입력 만들기)

  # 평가 세트 100문장 → B 조건 CSV (evaluate.py --pred 로 바로 넣는다)
  python scripts/predict_normalizer.py --input data/eval/eval_set.csv --out results/kobart_pred.csv
  python scripts/evaluate.py --eval data/eval/eval_set.csv --pred results/kobart_pred.csv --translate --model madlad --tag madlad

  # 화자 분리한 test 분할로 변환 성능만 보기 (평가 세트가 나오기 전에도 가능)
  python scripts/predict_normalizer.py --input data/pairs/test.csv --n 2000 --out results/norm_test_pred.csv

  # 문장 몇 개 직접 넣어 보기
  python scripts/predict_normalizer.py --text "니 밥 뭇나" "어데 가노"

입력 CSV 열: dialect (필수), id (없으면 행 번호), standard (있으면 점수까지 계산)
출력 CSV 열: id, dialect, pred_standard (+ standard, chrF)
"""
import argparse
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.aihub import norm_key  # noqa: E402
from src.normalize import Normalizer  # noqa: E402

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")


def scores(pred, ref):
    import jiwer
    import sacrebleu
    p = [norm_key(x) for x in pred]
    r = [norm_key(x) for x in ref]
    return {
        "chrF": round(sacrebleu.corpus_chrf(list(pred), [list(ref)]).score, 2),
        "BLEU_char": round(sacrebleu.corpus_bleu(list(pred), [list(ref)], tokenize="char").score, 2),
        "CER": round(jiwer.cer(r, p), 4),
        "exact": round(sum(a == b for a, b in zip(p, r)) / max(len(r), 1), 4),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default="checkpoints/kobart_norm/best")
    ap.add_argument("--input", help="dialect 열이 있는 CSV")
    ap.add_argument("--text", nargs="+", help="CSV 대신 문장을 직접 넣을 때")
    ap.add_argument("--out", default="results/kobart_pred.csv")
    ap.add_argument("--n", type=int, default=0, help="CSV에서 앞쪽이 아니라 무작위로 n개만 (0이면 전부)")
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--dtype", default="auto")
    args = ap.parse_args()
    if not args.input and not args.text:
        sys.exit("--input 또는 --text 중 하나는 필요")

    norm = Normalizer(args.ckpt, dtype=args.dtype)
    print(f"정규화 모델 {norm.name} ({norm.dtype}, {norm.device}), 접두어 {norm.prefix!r}")

    if args.text:
        preds, sec = norm.normalize(args.text, args.batch)
        for d, p in zip(args.text, preds):
            print(f"  사투리 {d}\n  표준어 {p}\n")
        print(f"{sec:.2f}초")
        return

    df = pd.read_csv(args.input).fillna("")
    if "dialect" not in df.columns:
        sys.exit(f"{args.input}에 'dialect' 열이 없음")
    if args.n and len(df) > args.n:
        df = df.sample(n=args.n, random_state=42)
    df = df.reset_index(drop=True)
    if "id" not in df.columns:
        df.insert(0, "id", [f"row{i:05d}" for i in range(len(df))])
    df["id"] = df["id"].astype(str)

    preds, sec = norm.normalize(df.dialect.tolist(), args.batch)
    df["pred_standard"] = preds
    print(f"{len(df)}문장 변환 {sec:.1f}초 (문장당 {sec / max(len(df), 1):.3f}초)")
    empty = sum(p == "" for p in preds)
    if empty:
        print(f"[!] 빈 출력 {empty}개. evaluate.py는 B 조건 칸이 하나라도 비면 B를 건너뛴다")

    cols = ["id", "dialect", "pred_standard"]
    if "standard" in df.columns:
        import sacrebleu
        df["chrF"] = [round(sacrebleu.sentence_chrf(p, [r]).score, 1) for p, r in zip(preds, df.standard)]
        cols += ["standard", "chrF"]
        m = scores(preds, df.standard.tolist())
        c = scores(df.dialect.tolist(), df.standard.tolist())
        print(f"\n모델     chrF {m['chrF']}  글자BLEU {m['BLEU_char']}  CER {m['CER']}  완전일치 {m['exact']}")
        print(f"복사 기준 chrF {c['chrF']}  글자BLEU {c['BLEU_char']}  CER {c['CER']}  완전일치 {c['exact']}")
        print("※ 모델이 복사 기준(입력을 그대로 낸 것)보다 높아야 실제로 고친 것이다.")
        worst = df.sort_values("chrF").head(5)
        print("\nchrF 낮은 문장 5개 (오류 분석용):")
        for r in worst.itertuples():
            print(f"  [{r.chrF}] 사투리 {r.dialect}\n         정답   {r.standard}\n         출력   {r.pred_standard}")

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    df[cols].to_csv(args.out, index=False, encoding="utf-8-sig")
    print(f"\n저장 → {args.out}")


if __name__ == "__main__":
    main()
