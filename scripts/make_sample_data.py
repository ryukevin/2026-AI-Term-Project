"""
공유용 샘플 데이터 만들기 (교수님·다른 PC에서 노트북을 데이터 없이 끝까지 돌려 볼 수 있게)

  python scripts/make_sample_data.py                    # train 5000 / valid 500 / test 500
  python scripts/make_sample_data.py --train 10000

data/pairs (전체 학습쌍)에서 분할별로 종류(일상대화, 노년층 3종) 비율을 유지하며 무작위로 뽑는다.
화자 기준 분할은 그대로 유지된다 (원래 분할 안에서만 뽑으므로).

출력: data_sample/pairs/{train,valid,test}.csv, filter_steps.csv, README.txt
※ AI Hub 데이터 일부다. 공유 허락 범위(제출용인지, 공개 저장소까지인지)를 지킬 것.
   기본으로 .gitignore에 들어 있어서 GitHub에는 올라가지 않는다.
"""
import argparse
import shutil
import sys
from pathlib import Path

import pandas as pd

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", default="data/pairs")
    ap.add_argument("--out", default="data_sample/pairs")
    ap.add_argument("--train", type=int, default=5000)
    ap.add_argument("--valid", type=int, default=500)
    ap.add_argument("--test", type=int, default=500)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    src, out = Path(args.src), Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    for sp, n in (("train", args.train), ("valid", args.valid), ("test", args.test)):
        df = pd.read_csv(src / f"{sp}.csv").fillna("")
        frac = min(1.0, n / len(df))
        smp = df.groupby("subset", group_keys=False).sample(frac=frac, random_state=args.seed)
        smp.sample(frac=1, random_state=args.seed).to_csv(out / f"{sp}.csv", index=False, encoding="utf-8-sig")
        print(f"{sp}: 전체 {len(df):,} → 샘플 {len(smp):,}  {smp.subset.value_counts().to_dict()}")
    if (src / "filter_steps.csv").exists():
        shutil.copy(src / "filter_steps.csv", out / "filter_steps.csv")
    (out.parent / "README.txt").write_text(
        "13조 인공지능 텀프로젝트 샘플 데이터\n"
        "출처: AI Hub 「한국어 방언 발화(경상도)」, 「중·노년층 한국어 방언 데이터(경상도)」 (과학기술정보통신부, 한국지능정보사회진흥원)\n"
        "전체 학습쌍 56만 개 중 일부를 무작위로 뽑은 것. 과제 확인·학습 시연 용도로만 사용.\n"
        "filter_steps.csv는 전체 데이터 기준 정제 단계별 개수.\n", encoding="utf-8")
    print(f"저장 → {out}")


if __name__ == "__main__":
    main()
