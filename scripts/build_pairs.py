"""
AI Hub 라벨 JSON → 방언/표준어 학습쌍 CSV (train/valid/test)

  python scripts/build_pairs.py                       # data/raw 아래 두 데이터셋 전부
  python scripts/build_pairs.py --src data/raw/dialect_gs --max_train 50000

필터 순서 (단계마다 남은 개수를 로그로 남긴다 → data/pairs/build_log.txt):
  1. 빈 값/손상(None, 시간 오류)       제거
  2. 개인정보 마스킹·알아들을 수 없음   제거  (&name1&, @이름1, #이름#, (()))
  3. 전사 기호 정리 후 빈 문장           제거  ({laughing}, -그-, ~ 등은 지우고 방언은 그대로)
  4. 방언 == 표준어                      제거  (복사 학습 방지)
  5. 방언 3어절 미만                     제거
  6. 길이 비율 이상 (손상 의심)          제거  (표준어/방언 글자 수 비율이 범위 밖)
  7. 중복 (방언, 표준어) 쌍              제거
  8. 화자(그룹) 단위 80/10/10 분리
       - 중·노년층: speakerId(전역 화자 ID) 기준
       - 014 방언 발화: 화자 ID가 파일 안에서만 유효 → 대화 파일 단위로 분리
  9. 누출 제거: valid/test 문장 중 train에 같은 방언 또는 표준어가 있으면 제거
       (따라말하기는 같은 대본을 여러 화자가 읽어서 이 단계가 꼭 필요)
 10. 평가 세트(data/eval/*.csv)와 겹치는 문장은 모든 분할에서 제거
"""
import argparse
import random
import sys
from collections import Counter, OrderedDict
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.aihub import clean_text, has_mask, iter_json, norm_key, parse  # noqa: E402

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", nargs="+", default=["data/raw/dialect_gs", "data/raw/elder_gs"])
    ap.add_argument("--out_dir", default="data/pairs")
    ap.add_argument("--eval_dir", default="data/eval", help="평가 세트 CSV 폴더 (dialect, standard 열)")
    ap.add_argument("--min_eojeol", type=int, default=3)
    ap.add_argument("--ratio_min", type=float, default=0.5)
    ap.add_argument("--ratio_max", type=float, default=2.0)
    ap.add_argument("--max_train", type=int, default=50000, help="train 샘플 파일 크기 (0이면 만들지 않음)")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--max_files", type=int, default=0, help="빠른 테스트용: 데이터셋마다 JSON 최대 개수")
    args = ap.parse_args()

    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    log_lines = []

    def log(msg=""):
        print(msg, flush=True)
        log_lines.append(msg)

    # ------------------------------------------------ 0. 읽기
    utts, n_json, n_bad = [], 0, 0
    for s in args.src:
        k = 0
        for src, name, d in iter_json(s):
            n_json += 1
            k += 1
            if d is None:
                n_bad += 1
                continue
            utts.extend(parse(src, name, d))
            if k % 2000 == 0:
                print(f"  {s}: JSON {k}개 읽음, 발화 누적 {len(utts)}", flush=True)
            if args.max_files and k >= args.max_files:
                break
    log(f"JSON {n_json}개 읽음 (파싱 실패 {n_bad}개)")

    df = pd.DataFrame([u.to_dict() for u in utts])
    steps = OrderedDict()

    def step(name, frame):
        steps[name] = frame.groupby("dataset").size().to_dict() | {"전체": len(frame)}
        return frame

    df = step("0. 원본 발화", df)

    # ------------------------------------------------ 1. 빈 값/손상
    ok = (df.dialect.str.strip() != "") & (df.standard.str.strip() != "") \
        & df.start.notna() & df.end.notna() & (df.end > df.start)
    df = step("1. 빈 값·시간 오류 제거", df[ok])

    # ------------------------------------------------ 2. 마스킹/불명확
    m = (df.dialect + " " + df.standard + " " + df.form).map(has_mask)
    df = step("2. 개인정보 마스킹·(()) 제거", df[~m])

    # ------------------------------------------------ 3. 전사 기호 정리
    df = df.assign(dialect_raw=df.dialect,
                   dialect=df.dialect.map(lambda t: clean_text(t, "dialect")),
                   standard=df.standard.map(lambda t: clean_text(t, "standard")))
    df = step("3. 기호 정리 후 빈 문장 제거", df[(df.dialect != "") & (df.standard != "")])

    # ------------------------------------------------ 4. 방언 == 표준어
    dk, sk = df.dialect.map(norm_key), df.standard.map(norm_key)
    df = step("4. 방언=표준어 쌍 제거", df[dk != sk])

    # ------------------------------------------------ 5. 짧은 문장
    df = df.assign(n_eojeol=df.dialect.str.split().map(len))
    df = step(f"5. {args.min_eojeol}어절 미만 제거", df[df.n_eojeol >= args.min_eojeol])

    # ------------------------------------------------ 6. 길이 비율
    ratio = df.standard.map(lambda t: len(norm_key(t))) / df.dialect.map(lambda t: max(1, len(norm_key(t))))
    df = step(f"6. 길이 비율 {args.ratio_min}~{args.ratio_max} 밖 제거",
              df[(ratio >= args.ratio_min) & (ratio <= args.ratio_max)])

    # ------------------------------------------------ 7. 중복
    df = step("7. 중복 쌍 제거", df.drop_duplicates(subset=["dialect", "standard"]))

    # ------------------------------------------------ 8. 화자(그룹) 단위 분리
    rng = random.Random(args.seed)
    split_of = {}
    for ds, sub in df.groupby("dataset"):
        groups = sorted(sub.group.unique())
        rng.shuffle(groups)
        n = len(groups)
        n_tr, n_va = int(n * 0.8), int(n * 0.1)
        for i, g in enumerate(groups):
            split_of[g] = "train" if i < n_tr else ("valid" if i < n_tr + n_va else "test")
        log(f"[{ds}] 분리 그룹 {n}개 → train {n_tr} / valid {n_va} / test {n - n_tr - n_va}")
    df = df.assign(split=df.group.map(split_of))

    # ------------------------------------------------ 9. 누출 제거
    df = df.assign(dk=df.dialect.map(norm_key), sk=df.standard.map(norm_key))
    tr = df[df.split == "train"]
    tr_d, tr_s = set(tr.dk), set(tr.sk)
    leak = (df.split != "train") & (df.dk.isin(tr_d) | df.sk.isin(tr_s))
    va = df[(df.split == "valid") & ~leak]
    leak |= (df.split == "test") & (df.dk.isin(set(va.dk)) | df.sk.isin(set(va.sk)))
    log(f"누출 제거: valid {int((leak & (df.split == 'valid')).sum())}개, test {int((leak & (df.split == 'test')).sum())}개")
    df = step("9. train과 겹치는 valid/test 문장 제거", df[~leak])

    # ------------------------------------------------ 10. 평가 세트 제외
    ev_files = sorted(Path(args.eval_dir).glob("*.csv")) if Path(args.eval_dir).exists() else []
    ev_keys = set()
    for f in ev_files:
        e = pd.read_csv(f)
        for col in ("dialect", "standard"):
            if col in e.columns:
                ev_keys |= set(e[col].dropna().map(norm_key))
    if ev_keys:
        hit = df.dk.isin(ev_keys) | df.sk.isin(ev_keys)
        log(f"평가 세트 {len(ev_files)}개 파일과 겹치는 문장 {int(hit.sum())}개 제거")
        df = step("10. 평가 세트와 겹치는 문장 제거", df[~hit])
    else:
        log("평가 세트 CSV가 아직 없음 → 나중에 data/eval/ 에 넣고 다시 실행할 것")

    # ------------------------------------------------ 저장
    cols = ["id", "dataset", "subset", "split", "group", "speaker", "file_id",
            "start", "end", "n_eojeol", "dialect", "standard"]
    df = df.reset_index(drop=True)
    df.insert(0, "id", [f"{d[:1]}{i:07d}" for i, d in enumerate(df.dataset)])
    for sp in ("train", "valid", "test"):
        part = df[df.split == sp][cols]
        part.to_csv(out / f"{sp}.csv", index=False, encoding="utf-8-sig")
    tr = df[df.split == "train"][cols]
    if args.max_train and len(tr) > args.max_train:
        tr.sample(n=args.max_train, random_state=args.seed) \
          .to_csv(out / f"train_{args.max_train // 1000}k.csv", index=False, encoding="utf-8-sig")
        log(f"train 샘플 {args.max_train}개 → train_{args.max_train // 1000}k.csv")

    # ------------------------------------------------ 로그
    log("\n=== 필터 단계별 남은 개수 ===")
    table = pd.DataFrame(steps).T.fillna(0).astype(int)
    log(table.to_string())
    log("\n=== 최종 분할 ===")
    log(df.pivot_table(index="dataset", columns="split", values="id", aggfunc="count",
                       margins=True, margins_name="합계").fillna(0).astype(int).to_string())
    log("\n=== 세부 종류(subset)별 ===")
    log(df.groupby(["dataset", "subset", "split"]).size().unstack(fill_value=0).to_string())
    log(f"\n방언 어절 수 평균 {df.n_eojeol.mean():.1f}, 중앙값 {df.n_eojeol.median():.0f}, 최대 {df.n_eojeol.max()}")

    log("\n=== 예시 (각 데이터셋 5개) ===")
    for ds, sub in df.groupby("dataset"):
        for _, r in sub.sample(n=min(5, len(sub)), random_state=1).iterrows():
            log(f"[{ds}/{r.subset}] 방언: {r.dialect}\n{'':>14}표준: {r.standard}")
    table.to_csv(out / "filter_steps.csv", encoding="utf-8-sig")
    (out / "build_log.txt").write_text("\n".join(log_lines), encoding="utf-8")
    print(f"\n저장: {out}/train.csv, valid.csv, test.csv, build_log.txt, filter_steps.csv")


if __name__ == "__main__":
    main()
