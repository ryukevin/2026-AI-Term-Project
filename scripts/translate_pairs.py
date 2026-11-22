"""
학습쌍(사투리-표준어)을 실제로 번역해서 나란히 보는 확인용 스크립트. GPU PC용.

  python scripts/translate_pairs.py                      # test 분할에서 종류별 25쌍 = 100쌍
  python scripts/translate_pairs.py --per_subset 50 --model madlad

같은 문장을 세 가지로 번역한다 (번역 모델·디코딩 설정은 evaluate.py와 같음)
  A  사투리 원문 그대로 번역
  B  우리 정규화 모델이 바꾼 표준어를 번역   (체크포인트가 있을 때만)
  C  정답 표준어를 번역

중국어 정답이 없으므로 진짜 점수는 아니다. 대신 "C 번역과 얼마나 비슷한가"(chrF)를 참고값으로 준다.
B가 A보다 C에 가까우면 정규화가 번역을 표준어 쪽으로 끌어온다는 뜻이다. 진짜 점수는 평가 세트로 낸다.

출력: results/번역확인_test.xlsx  (AI Hub 문장 포함 → 커밋 금지)
"""
import argparse
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.translate import DECODE, Translator  # noqa: E402

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

SUBSET_NAME = {"conv": "014 일상대화", "st": "노년층 따라말하기", "say": "노년층 질문답하기", "talk": "노년층 2인대화"}
CKPT_CANDIDATES = ["checkpoints/kobart_norm/best", "checkpoints/smoke/best"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pairs", default="data/pairs/test.csv")
    ap.add_argument("--per_subset", type=int, default=25)
    ap.add_argument("--model", default="madlad", help="번역 모델 (평가와 같은 것)")
    ap.add_argument("--tgt", default="zh")
    ap.add_argument("--ckpt", default="", help="정규화 모델. 생략하면 kobart_norm → smoke 순서로 찾음")
    ap.add_argument("--out", default="results/번역확인_test.xlsx")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    df = pd.read_csv(args.pairs).fillna("")
    df = pd.concat([g.sample(n=min(args.per_subset, len(g)), random_state=args.seed)
                    for _, g in df.groupby("subset")], ignore_index=True)
    df["종류"] = df.subset.map(lambda x: SUBSET_NAME.get(x, x))
    print(f"{args.pairs}에서 {len(df)}쌍 뽑음 (종류별 최대 {args.per_subset})")

    # ---------- B: 정규화 (번역 모델 올리기 전에 끝내고 메모리 비움)
    ckpt = args.ckpt or next((c for c in CKPT_CANDIDATES if Path(c, "config.json").exists()), "")
    if ckpt:
        import torch
        from src.normalize import Normalizer
        norm = Normalizer(ckpt)
        df["pred_standard"], sec = norm.normalize(df.dialect.tolist())
        print(f"정규화 {ckpt}: {sec:.1f}초")
        del norm
        torch.cuda.empty_cache()
    else:
        print("[!] 정규화 모델 체크포인트가 없어서 B 조건은 건너뜀")

    # ---------- 번역
    tr = Translator(args.model)
    print(f"번역 모델 {tr.name} ({tr.dtype}), 디코딩 {DECODE}")
    conds = [("A", "dialect"), ("C", "standard")] + ([("B", "pred_standard")] if ckpt else [])
    for c, col in conds:
        df[f"zh_{c}"], sec = tr.translate(df[col].tolist(), args.tgt)
        print(f"  {c} 번역 {sec:.1f}초")

    # ---------- 참고값: C 번역 대비 chrF
    import sacrebleu
    ref = df.zh_C.tolist()
    df["A≈C"] = [round(sacrebleu.sentence_chrf(h, [r]).score, 1) for h, r in zip(df.zh_A, ref)]
    print("\n=== 참고: C(정답 표준어) 번역과의 chrF (진짜 점수 아님) ===")
    summ = {"A 사투리 그대로": round(sacrebleu.corpus_chrf(df.zh_A.tolist(), [ref]).score, 2)}
    if ckpt:
        df["B≈C"] = [round(sacrebleu.sentence_chrf(h, [r]).score, 1) for h, r in zip(df.zh_B, ref)]
        summ["B 모델 표준어"] = round(sacrebleu.corpus_chrf(df.zh_B.tolist(), [ref]).score, 2)
    for k, v in summ.items():
        print(f"  {k}: {v}")
    by = df.groupby("종류")[[c for c in ("A≈C", "B≈C") if c in df]].mean().round(1)
    print(by.to_string())

    cols = {"id": "ID", "종류": "종류", "dialect": "사투리", "standard": "정답 표준어"}
    if ckpt:
        cols["pred_standard"] = "모델 표준어"
    cols["zh_A"] = "A 사투리→중국어"
    if ckpt:
        cols["zh_B"] = "B 모델표준어→중국어"
    cols["zh_C"] = "C 정답표준어→중국어"
    cols["A≈C"] = "A≈C"
    if ckpt:
        cols["B≈C"] = "B≈C"
    out = df[list(cols)].rename(columns=cols)

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    try:
        from openpyxl.styles import Alignment, Font
        with pd.ExcelWriter(args.out, engine="openpyxl") as xw:
            out.to_excel(xw, sheet_name="문장별", index=False)
            ws = xw.sheets["문장별"]
            for i, c in enumerate(out.columns, 1):
                ws.column_dimensions[ws.cell(1, i).column_letter].width = 8 if "≈" in c else (12 if c in ("ID", "종류") else 40)
                ws.cell(1, i).font = Font(bold=True)
            for row in ws.iter_rows(min_row=2):
                for cell in row:
                    cell.alignment = Alignment(wrap_text=True, vertical="top")
            ws.freeze_panes = "A2"
            s = pd.DataFrame({"조건": list(summ), "C 번역 대비 chrF (참고)": list(summ.values())})
            s.to_excel(xw, sheet_name="요약", index=False)
            by.to_excel(xw, sheet_name="요약", startrow=len(s) + 3)
            xw.sheets["요약"].column_dimensions["A"].width = 22
        print(f"\n저장 → {args.out}")
    except ImportError:
        csv = Path(args.out).with_suffix(".csv")
        out.to_csv(csv, index=False, encoding="utf-8-sig")
        print(f"\n[!] openpyxl이 없어서 CSV로 저장 → {csv}  (pip install openpyxl 하면 엑셀로 나옴)")


if __name__ == "__main__":
    main()
