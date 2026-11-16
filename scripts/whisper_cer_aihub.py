"""
AI Hub 방언 데이터로 Whisper CER 측정 (13조 1주차)

라벨 JSON의 발화 구간(start/end)만큼 긴 음성을 잘라 Whisper로 전사하고 CER을 계산한다.
라벨과 음성 모두 zip째로 넣어도 된다 (음성 zip을 풀 필요 없음).
두 데이터셋(014 방언 발화, 139-1 중·노년층) 모두 읽는다.
음성은 일부만 받아도 된다. 음성이 있는 라벨만 골라서 측정한다.

1) 구조 확인 (음성 없이도 동작):
  python scripts/whisper_cer_aihub.py --label_dir data/raw/dialect_gs --audio_dir data/raw/audio --inspect

2) 실행 (사투리가 들어간 발화 30개 무작위):
  python scripts/whisper_cer_aihub.py --label_dir data/raw/dialect_gs --audio_dir data/raw/audio --n 30

결과: results/whisper_cer_<이름>.csv, results/whisper_cer_summary.csv 에 한 줄 추가
"""
import argparse
import io
import json
import random
import sys
import time
import zipfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.aihub import (clean_text, detect_format, has_mask, iter_json, list_json_stems,  # noqa: E402
                       norm_key, parse)

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")


def cer(ref: str, hyp: str) -> float:
    import jiwer
    r, h = norm_key(ref), norm_key(hyp)
    return jiwer.cer(r, h) if r else float("nan")


# ---------- 음성 찾기/읽기 (폴더의 wav + zip 안의 wav)
def build_audio_index(dirs) -> dict:
    """{파일이름(확장자 없음): (경로, zip 안 이름 또는 None)}"""
    idx = {}
    for a in dirs:
        p = Path(a)
        if not p.exists():
            continue
        files = [p] if p.is_file() else p.rglob("*")
        for f in files:
            suf = f.suffix.lower()
            if suf == ".wav":
                idx[f.stem] = (str(f), None)
            elif suf == ".zip":
                try:
                    with zipfile.ZipFile(f) as z:
                        for n in z.namelist():
                            if n.lower().endswith(".wav"):
                                idx[Path(n).stem] = (str(f), n)
                except zipfile.BadZipFile:
                    print(f"  손상된 zip 건너뜀: {f}")
    return idx


_zip_cache = {}


def read_segment(loc, start: float, end: float):
    """발화 구간만 잘라서 16kHz mono float32 로 돌려준다."""
    import librosa
    import soundfile as sf

    path, member = loc
    if member is None:
        src = path
    else:                                   # 같은 wav를 여러 번 자를 수 있어서 마지막 1개만 캐시
        key = (path, member)
        if key not in _zip_cache:
            _zip_cache.clear()
            with zipfile.ZipFile(path) as z:
                _zip_cache[key] = z.read(member)
        src = io.BytesIO(_zip_cache[key])
    with sf.SoundFile(src) as f:
        sr = f.samplerate
        f.seek(int(start * sr))
        audio = f.read(int((end - start) * sr), dtype="float32")
    if audio.ndim > 1:
        audio = audio.mean(axis=1)
    if sr != 16000:
        audio = librosa.resample(audio, orig_sr=sr, target_sr=16000)
    return audio


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
    ap.add_argument("--dry_run", action="store_true", help="Whisper 없이 음성 자르기만 시험")
    args = ap.parse_args()

    audio_index = build_audio_index(args.audio_dir)
    n_zip = sum(1 for _, m in audio_index.values() if m)
    print(f"음성 WAV {len(audio_index)}개 발견 (zip 안 {n_zip}개) ({', '.join(args.audio_dir)})")
    if audio_index:
        print("  음성 이름 예시:", list(audio_index)[:3])

    # ---------------- 구조 확인
    if args.inspect:
        for ld in args.label_dir:                  # 라벨 폴더마다 첫 JSON 하나씩 구조 출력
            for src, name, d in iter_json(ld):
                if d is None:
                    continue
                utts = parse(src, name, d)
                print(f"\n[예시] {src} :: {name}  (형식 {detect_format(d)})")
                print("  최상위 키:", list(d.keys()))
                print(f"  발화 {len(utts)}개, 첫 3개:")
                for u in utts[:3]:
                    print("   ", json.dumps(u.to_dict(), ensure_ascii=False))
                break
        for ld in args.label_dir:                  # 이름 매칭은 zip 목록만 보고 빠르게 센다
            stems = list_json_stems(ld)
            n_match = sum(s in audio_index for s in stems)
            print(f"\n{ld}: 라벨 JSON {len(stems)}개 중 음성 있는 것 {n_match}개")
        if not audio_index:
            print("→ 음성이 없다. AI Hub에서 '원천데이터'(wav)를 받아 --audio_dir 로 지정해야 CER 측정 가능.")
        elif not any(s in audio_index for ld in args.label_dir for s in list_json_stems(ld)):
            print("→ 음성은 있는데 라벨과 이름이 하나도 안 맞는다. 위 '음성 이름 예시'와 JSON 이름을 비교할 것.")
        return

    # ---------------- 후보 발화 모으기 (음성이 있는 JSON만 연다)
    candidates, n_json = [], 0
    for src, name, d in iter_json(args.label_dir, only=set(audio_index)):
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
    picks.sort(key=lambda x: (x[0][0], x[0][1] or "", x[1].start))  # 같은 음성끼리 모아 zip 읽기 줄이기

    if args.dry_run:      # 모델 없이 음성 자르기만 확인
        for i, (loc, u, ref, std) in enumerate(picks, 1):
            audio = read_segment(loc, u.start, u.end)
            print(f"[{i}/{len(picks)}] {u.file_id} {u.start:.2f}~{u.end:.2f}s → {len(audio) / 16000:.2f}s 잘림 | {ref}")
        print("dry_run 끝: 위 길이가 (end-start)와 비슷하면 음성 매칭 정상")
        return

    import pandas as pd
    import torch
    from transformers import pipeline

    gpu = torch.cuda.is_available()
    print(f"모델 {args.model} 로딩 ({'GPU' if gpu else 'CPU'})")
    asr = pipeline("automatic-speech-recognition", model=args.model,
                   torch_dtype=torch.float16 if gpu else torch.float32,
                   device=0 if gpu else -1)

    rows = []
    for i, (loc, u, ref, std) in enumerate(picks, 1):
        audio = read_segment(loc, u.start, u.end)
        if len(audio) < 1600:
            print(f"[{i}/{len(picks)}] 음성 구간이 비어 있음 → 건너뜀 ({u.file_id} {u.start}~{u.end})")
            continue
        t0 = time.time()
        hyp = asr({"raw": audio, "sampling_rate": 16000},
                  generate_kwargs={"language": "korean", "task": "transcribe"})["text"].strip()
        row = {"dataset": u.dataset, "file": u.file_id, "start": u.start, "end": u.end,
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
