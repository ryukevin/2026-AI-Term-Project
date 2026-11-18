"""
13조 1주차 Whisper CER 측정을 순서대로 실행 (음성이 있는 PC에서)

  python scripts/run_whisper.py            # 또는 scripts\run_whisper.bat 더블클릭
  python scripts/run_whisper.py --n 50     # 데이터셋마다 50문장

순서: 환경 점검 → 이름 매칭 확인 → 음성 자르기 시험 → 014 CER → 중·노년층 CER
결과: results/whisper_cer_dialect.csv, whisper_cer_elder.csv, whisper_cer_summary.csv,
      whisper_log_dialect.txt, whisper_log_elder.txt
"""
import argparse
import os
import subprocess
import sys
from pathlib import Path

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

ROOT = Path(__file__).resolve().parents[1]
PY = [sys.executable, "-X", "utf8"]
W = "scripts/whisper_cer_aihub.py"
LABELS = ["data/raw/dialect_gs", "data/raw/elder_gs"]
AUDIO = ["--audio_dir", "data/raw/audio"]


def run(title, args, log=None):
    print(f"\n{'=' * 60}\n{title}\n{'=' * 60}", flush=True)
    env = {**os.environ, "PYTHONUTF8": "1"}
    if log is None:
        return subprocess.run(PY + args, cwd=ROOT, env=env).returncode
    # 화면에도 보여 주고 파일에도 남긴다
    with open(ROOT / log, "w", encoding="utf-8") as f:
        p = subprocess.Popen(PY + args, cwd=ROOT, env=env, stdout=subprocess.PIPE,
                             stderr=subprocess.STDOUT, text=True, encoding="utf-8", errors="replace")
        for line in p.stdout:
            print(line, end="", flush=True)
            f.write(line)
        return p.wait()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=30)
    ap.add_argument("--skip_check", action="store_true", help="환경 점검 건너뛰기")
    args = ap.parse_args()
    (ROOT / "results").mkdir(exist_ok=True)

    if not args.skip_check and run("[1/5] 환경 점검", ["scripts/check_env.py"]) != 0:
        print("\n환경 점검에서 문제가 있습니다. 위 안내대로 고친 뒤 다시 실행하세요.")
        return 1
    run("[2/5] 라벨-음성 이름 매칭 확인", [W, "--label_dir", *LABELS, *AUDIO, "--inspect"])
    run("[3/5] 모델 없이 음성 자르기 시험 (5개)", [W, "--label_dir", *LABELS, *AUDIO, "--n", "5", "--dry_run"])
    rc1 = run(f"[4/5] Whisper CER - 014 방언 발화 ({args.n}개)",
              [W, "--label_dir", LABELS[0], *AUDIO, "--n", str(args.n), "--tag", "dialect"],
              log="results/whisper_log_dialect.txt")
    rc2 = run(f"[5/5] Whisper CER - 중·노년층 ({args.n}개)",
              [W, "--label_dir", LABELS[1], *AUDIO, "--n", str(args.n), "--tag", "elder"],
              log="results/whisper_log_elder.txt")

    print(f"\n{'=' * 60}")
    summary = ROOT / "results" / "whisper_cer_summary.csv"
    if summary.exists():
        print(summary.read_text(encoding="utf-8-sig"))
    print("끝. results 폴더의 whisper_cer_*.csv, whisper_log_*.txt 를 GitHub에 올리거나 Kevin에게 보내 주세요.")
    print("음성 파일(wav, zip)은 올리지 말 것. 한쪽 데이터셋 음성만 있으면 다른 쪽은 '후보 0개'가 정상.")
    if rc1 or rc2:
        print(f"[!] 에러가 있었음 (014: {rc1}, 중·노년층: {rc2}) → whisper_log_*.txt 확인")
    return 0


if __name__ == "__main__":
    sys.exit(main())
