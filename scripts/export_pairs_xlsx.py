"""
학습쌍 CSV → 엑셀 정리본 (발표·보고서·팀 공유용). GPU 필요 없음.

  python scripts/export_pairs_xlsx.py                 # 요약 + 종류별 샘플
  python scripts/export_pairs_xlsx.py --per_subset 500
  python scripts/export_pairs_xlsx.py --all           # 분할별 전체 (train 46만 행, 파일이 큼)

출력: results/데이터쌍_정리.xlsx
  요약        : 최종 분할 개수, 필터 단계별 남은 개수
  종류별 샘플 : 014 일상대화 / 노년층 따라말하기·질문답하기·2인대화에서 무작위 n쌍
                "바뀐 부분" 열에 사투리 → 표준어로 달라진 어절만 표시
※ AI Hub 문장이 들어 있으므로 GitHub에 올리지 않는다 (results/는 커밋 대상 아님).
"""
import argparse
import difflib
import sys
from pathlib import Path

import pandas as pd

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

SUBSET_NAME = {
    ("dialect_gs", "conv"): "014 일상대화",
    ("elder_gs", "st"): "노년층 따라말하기",
    ("elder_gs", "say"): "노년층 질문답하기",
    ("elder_gs", "talk"): "노년층 2인대화",
}
DATASET_NAME = {"dialect_gs": "014 한국어 방언 발화(경상도)", "elder_gs": "139-1 중·노년층 방언(경상도)"}
COLS = {"id": "ID", "dataset": "데이터셋", "subset": "종류", "split": "분할", "n_eojeol": "어절 수",
        "dialect": "사투리", "standard": "표준어", "changes": "바뀐 부분"}


def changes(d: str, s: str) -> str:
    """어절 단위로 달라진 곳만 '사투리 → 표준어'로 표시"""
    a, b = str(d).split(), str(s).split()
    out = []
    for op, i1, i2, j1, j2 in difflib.SequenceMatcher(None, a, b).get_opcodes():
        if op != "equal":
            out.append(f"{' '.join(a[i1:i2]) or '∅'} → {' '.join(b[j1:j2]) or '∅'}")
    return " / ".join(out)


def tidy(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["changes"] = [changes(d, s) for d, s in zip(df.dialect, df.standard)]
    df["subset"] = [SUBSET_NAME.get((a, b), b) for a, b in zip(df.dataset, df.subset)]
    df["dataset"] = df.dataset.map(lambda x: DATASET_NAME.get(x, x))
    return df[list(COLS)].rename(columns=COLS)


def fit_columns(ws, widths):
    from openpyxl.styles import Alignment, Font
    for i, w in enumerate(widths, 1):
        ws.column_dimensions[ws.cell(1, i).column_letter].width = w
    for c in ws[1]:
        c.font = Font(bold=True)
    for row in ws.iter_rows(min_row=2):
        for c in row:
            c.alignment = Alignment(wrap_text=True, vertical="top")
    ws.freeze_panes = "A2"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pairs_dir", default="data/pairs")
    ap.add_argument("--out", default="results/데이터쌍_정리.xlsx")
    ap.add_argument("--per_subset", type=int, default=300, help="종류별 샘플 수 (train에서)")
    ap.add_argument("--all", action="store_true", help="분할별 전체 시트도 넣기")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    pdir = Path(args.pairs_dir)
    parts = {sp: pd.read_csv(pdir / f"{sp}.csv").fillna("") for sp in ("train", "valid", "test")}
    allp = pd.concat(parts.values(), ignore_index=True)
    print(f"학습쌍 {len(allp)}개 읽음")

    # ---------- 요약
    split_tab = allp.assign(데이터=[SUBSET_NAME.get((a, b), b) for a, b in zip(allp.dataset, allp.subset)]) \
        .pivot_table(index="데이터", columns="split", values="id", aggfunc="count", fill_value=0)
    split_tab = split_tab[["train", "valid", "test"]].rename(columns={"train": "학습", "valid": "검증", "test": "시험"})
    split_tab["합계"] = split_tab.sum(axis=1)
    split_tab.loc["합계"] = split_tab.sum()
    steps = None
    if (pdir / "filter_steps.csv").exists():
        steps = pd.read_csv(pdir / "filter_steps.csv", index_col=0)
        steps = steps.rename(columns={"dialect_gs": "014 일상대화", "elder_gs": "139-1 중·노년층"})

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    with pd.ExcelWriter(args.out, engine="openpyxl") as xw:
        split_tab.to_excel(xw, sheet_name="요약", startrow=1)
        ws = xw.sheets["요약"]
        ws.cell(1, 1, "최종 학습쌍 (화자 기준 분할)")
        if steps is not None:
            r0 = len(split_tab) + 5
            ws.cell(r0, 1, "정제 단계별 남은 발화 수")
            steps.to_excel(xw, sheet_name="요약", startrow=r0)
        ws.column_dimensions["A"].width = 34
        for col in "BCDE":
            ws.column_dimensions[col].width = 14

        for (ds, sub), name in SUBSET_NAME.items():
            src = parts["train"][(parts["train"].dataset == ds) & (parts["train"].subset == sub)]
            if src.empty:
                continue
            smp = tidy(src.sample(n=min(args.per_subset, len(src)), random_state=args.seed))
            smp.to_excel(xw, sheet_name=name, index=False)
            fit_columns(xw.sheets[name], [10, 26, 16, 7, 8, 55, 55, 40])
            print(f"  {name}: {len(smp)}쌍")

        if args.all:
            for sp, df in parts.items():
                t = tidy(df)
                t.to_excel(xw, sheet_name=f"전체_{sp}", index=False)
                print(f"  전체_{sp}: {len(t)}쌍")
    print(f"저장 → {args.out}")


if __name__ == "__main__":
    main()
