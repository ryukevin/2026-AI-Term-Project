"""
AI Hub 방언 데이터로 Whisper CER 측정 (13조 1주차)

라벨 JSON의 발화 구간(start/end)만큼 긴 음성을 잘라 Whisper로 전사하고 CER을 계산한다.
라벨은 zip째로 넣어도 된다. 두 데이터셋(014 방언 발화, 139-1 중·노년층) 모두 읽는다.

1) 구조 확인 (음성 없이도 동작):
  python scripts/whisper_cer_aihub.py --label_dir data/raw/dialect_gs --audio_dir data/raw/audio --inspect

2) 실행 (사투리가 들어간 발화 30개 무작위):
  python scripts/whisper_cer_aihub.py --label_dir data/raw/dialect_gs --audio_dir data/raw/audio --n 30

결과: results/whisper_cer_<이름>.csv, results/whisper_cer_summary.csv 에 한 줄 추가
"""
import argparse
import json
import random
import re
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.aihub import clean_text, detect_format, has_mask, iter_json, norm_key, parse  # noqa: E402

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")


def cer(ref: str, hyp: str) -> float:
    import jiwer
    r, h = norm_key(ref), norm_key(hyp)
    return jiwer.cer(r, h) if r else float("nan")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--label_dir", nargs="+", required=True, help="라벨 폴더/zip/json (여러 개 가능)")
    ap.add_argument("--audio_dir", nargs="+", default=["data/raw/audio"], help="wav가 있는 폴더 (하위폴더 포함)")
    ap.add_argument("--n", type=int, default=30)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--all_utt", action="store_true", help="사투리 없는 발화도 포함")
    ap.add_argument("--min_sec", type=float, default=1.5)
    ap.add_argument("--max_sec", type=float, default=20.0)
    ap.add_argument("--max_files", type=int, default=0, help="빠른 테스트용: 라벨 JSON 최대 개수")
    ap.add_argument("--model", default="openai/whisper-large-v3-turbo")
    ap.add_argument("--tag", default="", help="결과 파일 이름 꼬리표 (예: dialect, elder)")
    ap.add_argument("--inspect", action="store_true")
    args = ap.parse_args()

    audio_index = {}
    for a in args.audio_dir:
        if Path(a).exists():
            audio_index.update({p.stem: p for p in Path(a).rglob("*.wav")})
    print(f"음성 WAV {len(audio_index)}개 발견 ({', '.join(args.audio_dir)})")

    # ---------------- 구조 확인
    if args.inspect:
        seen_fmt, n_json, n_bad, n_match = set(), 0, 0, 0
        for src, name, d in iter_json(args.label_dir):
            n_json += 1
            if d is None:
                n_bad += 1
                continue
            stem = Path(name).stem
            n_match += stem in audio_index
            fmt = detect_format(d)
            if fmt not in seen_fmt:
                seen_fmt.add(fmt)
                utts = parse(src, name, d)
                print(f"\n[예시] {src} :: {name}  (형식 {fmt})")
                print("  최상위 키:", list(d.keys()))
                print(f"  발화 {len(utts)}개, 첫 3개:")
                for u in utts[:3]:
                    print("   ", json.dumps(u.to_dict(), ensure_ascii=False))
                print("  같은 이름 WAV 있음?", stem in audio_index)
            if args.max_files and n_json >= args.max_files:
                break
        print(f"\nJSON {n_json}개 (파싱 실패 {n_bad}개), 이름이 같은 WAV 있는 JSON {n_match}개")
        if not audio_index:
            print("→ 음성이 없다. AI Hub에서 '원천데이터'(wav)를 받아 --audio_dir 로 지정해야 CER 측정 가능.")
        return

    # ---------------- 후보 발화 모으기 (음성이 있는 JSON만)
    candidates, n_json = [], 0
    for src, name, d in iter_json(args.label_dir):
        n_json += 1
        if args.max_files and n_json > args.max_files:
            break
        wav = audio_index.get(Path(name).stem)
        if wav is None or d is None:
            continue
        for u in parse(src, name, d):
            if not (args.min_sec <= u.end - u.start <= args.max_sec):
                continue
            if not u.dialect or has_mask(u.dialect + " " + u.form):
                continue
            ref, std = clean_text(u.dialect, "dialect"), clean_text(u.standard, "standard")
            if not ref:
                continue
            if not args.all_utt and norm_key(ref) == norm_key(std):
                continue
            candidates.append((wav, u, ref, std))

    print(f"조건 맞는 발화 {len(candidates)}개")
    if not candidates:
        print("후보 0개 → --inspect로 라벨/음성 파일 이름이 맞는지 확인")
        return
    random.seed(args.seed)
    picks = random.sample(candidates, min(args.n, len(candidates)))

    import librosa
    import pandas as pd
    import soundfile as sf
    import torch
    from transformers import pipeline

    gpu = torch.cuda.is_available()
    print(f"모델 {args.model} 로딩 ({'GPU' if gpu else 'CPU'})")
    asr = pipeline("automatic-speech-recognition", model=args.model,
                   torch_dtype=torch.float16 if gpu else torch.float32,
                   device=0 if gpu else -1)

    rows = []
    for i, (wav, u, ref, std) in enumerate(picks, 1):
        info = sf.info(str(wav))
        audio, sr = sf.read(str(wav), start=int(u.start * info.samplerate),
                            stop=int(u.end * info.samplerate), dtype="float32")
        if audio.ndim > 1:
            audio = audio.mean(axis=1)
        if sr != 16000:
            audio = librosa.resample(audio, orig_sr=sr, target_sr=16000)
        t0 = time.time()
        hyp = asr({"raw": audio, "sampling_rate": 16000},
                  generate_kwargs={"language": "korean", "task": "transcribe"})["text"].strip()
        row = {"dataset": u.dataset, "file": wav.name, "start": u.start, "end": u.end,
               "dialect": ref, "standard": std, "whisper": hyp,
               "cer_vs_dialect": round(cer(ref, hyp), 4),
               "cer_vs_standard": round(cer(std, hyp), 4) if std else None,
               "sec": round(time.time() - t0, 2)}
        rows.append(row)
        print(f"[{i}/{len(picks)}] CER {row['cer_vs_dialect']:.3f}\n  정답: {ref}\n  전사: {hyp}")

    out_dir = Path("results")
    out_dir.mkdir(exist_ok=True)
    tag = f"_{args.tag}" if args.tag else ""
    res = pd.DataFrame(rows)
    res.to_csv(out_dir / f"whisper_cer{tag}.csv", index=False, encoding="utf-8-sig")

    closer = int((res["cer_vs_standard"] < res["cer_vs_dialect"]).sum())
    summary = {"tag": args.tag or "default", "model": args.model, "n": len(res),
               "cer_vs_dialect": round(res["cer_vs_dialect"].mean(), 4),
               "cer_vs_standard": round(res["cer_vs_standard"].mean(), 4),
               "closer_to_standard": f"{closer}/{len(res)}",
               "sec_per_utt": round(res["sec"].mean(), 2)}
    sp = out_dir / "whisper_cer_summary.csv"
    pd.concat([pd.read_csv(sp), pd.DataFrame([summary])] if sp.exists() else [pd.DataFrame([summary])]) \
        .to_csv(sp, index=False, encoding="utf-8-sig")

    print("\n===== 결과 =====")
    for k, v in summary.items():
        print(f"{k}: {v}")
    print("판단 기준(잠정): CER ≤ 0.15 → 음성 기능 포함 / > 0.30 → 텍스트 데모만")
    print(f"저장: {out_dir / f'whisper_cer{tag}.csv'}, {sp}")


if __name__ == "__main__":
    main()
