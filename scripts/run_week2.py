"""
2주차 기본 실험을 한 번에 돌린다. Kevin PC(RTX 3060 Ti)용.

  scripts\\run_week2.bat      (더블클릭)
  python scripts/run_week2.py [--retrain] [--max_minutes 30]

하는 일
  1) CUDA GPU 확인
  2) 정규화 모델 본 학습 (train_50k.csv, 5에폭, early stopping)
     - 학습이 끝까지 끝났으면 건너뜀 (--retrain 이면 처음부터 다시 학습)
     - 나눠서 학습: --max_minutes 30 이면 30분 학습 후 저장하고 멈춤. Ctrl+C로 멈춰도 저장됨.
       다시 실행하면 멈춘 곳부터 이어서 하고, 학습이 다 끝난 실행에서만 3), 4)로 넘어간다.
     - 처음 학습이 GPU 메모리 부족으로 실패하면 --batch 8 --grad_accum 2 로 한 번 더
  3) 화자 분리한 test 분할 2000문장으로 변환 점수 (모델 vs 입력 복사 기준)
  4) test 100쌍을 A(사투리)·B(모델 표준어)·C(정답 표준어) 세 가지로 중국어 번역해서 엑셀로
  (시작할 때 학습쌍 요약 엑셀도 만든다)

출력
  checkpoints/kobart_norm/best/          학습된 모델
  checkpoints/kobart_norm/train_log.csv  평가 시점별 점수
  results/norm_test_pred.csv             test 문장별 변환 결과 (AI Hub 문장 → 커밋 금지)
  results/데이터쌍_정리.xlsx              학습쌍 개수 요약 + 종류별 샘플
  results/번역확인_test.xlsx              A/B/C 번역 나란히 (참고 점수 포함)
  results/week2_log.txt                  이 실행의 전체 출력
"""
import subprocess
import sys
from datetime import datetime
from pathlib import Path

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

ROOT = Path(__file__).resolve().parents[1]
CKPT = Path("checkpoints/kobart_norm")
LOG = ROOT / "results" / "week2_log.txt"


def say(msg, log):
    print(msg, flush=True)
    log.write(msg + "\n")
    log.flush()


def run(cmd, log):
    """출력을 화면과 로그 파일에 동시에 쓴다."""
    say(f"\n$ {' '.join(cmd)}", log)
    p = subprocess.Popen(cmd, cwd=ROOT, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                         text=True, encoding="utf-8", errors="replace",
                         env={**__import__("os").environ, "PYTHONUTF8": "1", "PYTHONUNBUFFERED": "1"})
    try:
        for line in p.stdout:
            print(line, end="", flush=True)
            log.write(line)
            log.flush()
    except KeyboardInterrupt:
        # Ctrl+C는 학습 프로세스에도 같이 간다. 학습 쪽이 저장을 끝낼 때까지 출력을 계속 받는다
        say("\n[!] 멈춤 요청 → 학습이 저장을 마칠 때까지 기다림 (창을 닫지 말 것)", log)
        for line in p.stdout:
            print(line, end="", flush=True)
            log.write(line)
            log.flush()
    return p.wait()


def arg_value(name, default=None):
    if name in sys.argv:
        i = sys.argv.index(name)
        if i + 1 < len(sys.argv):
            return sys.argv[i + 1]
    return default


def main() -> int:
    retrain = "--retrain" in sys.argv
    max_minutes = arg_value("--max_minutes")
    LOG.parent.mkdir(exist_ok=True)
    # 나눠서 돌리면 실행마다 로그가 쌓이도록 이어 쓴다
    with open(LOG, "a", encoding="utf-8") as log:
        say(f"=== 2주차 실험 시작 {datetime.now():%Y-%m-%d %H:%M} ===", log)
        try:
            import torch
        except ImportError:
            say("[X] torch가 없음 → README의 '설치' 순서대로 설치", log)
            return 1
        if not torch.cuda.is_available():
            say(f"[X] CUDA GPU를 못 찾음 (torch {torch.__version__}). 3060 Ti PC에서 실행할 것.", log)
            return 1
        p = torch.cuda.get_device_properties(0)
        say(f"[O] GPU {p.name}, VRAM {p.total_memory / 2**30:.1f}GB", log)
        for f in ("data/pairs/train_50k.csv", "data/pairs/valid.csv", "data/pairs/test.csv"):
            if not (ROOT / f).exists():
                say(f"[X] {f} 없음 → python scripts/build_pairs.py 먼저 실행", log)
                return 1

        # ---------------- 0. 엑셀용 패키지 + 학습쌍 요약 엑셀
        try:
            import openpyxl  # noqa: F401
        except ImportError:
            say("\n[!] openpyxl 없음 → 설치", log)
            run([sys.executable, "-m", "pip", "install", "openpyxl"], log)
        say("\n=== 0. 학습쌍 요약 엑셀 ===", log)
        if (ROOT / "results" / "데이터쌍_정리.xlsx").exists() and not retrain:
            say("[O] results/데이터쌍_정리.xlsx 이미 있음 → 건너뜀", log)
        elif run([sys.executable, "scripts/export_pairs_xlsx.py"], log) != 0:
            say("[!] 엑셀 만들기 실패 (학습은 계속 진행)", log)

        # ---------------- 1. 학습
        ck = ROOT / CKPT
        done = (ck / "train_done.json").exists()
        has_save = any((ck / d / "train_state.pt").exists() for d in ("last", "last_old"))
        # 이어서 학습 기능 전에 끝까지 학습한 모델 (done 표시 없이 best만 있음)
        legacy = (ck / "best" / "norm_config.json").exists() and not done and not has_save
        train = [sys.executable, "scripts/train_normalizer.py", "--out", str(CKPT)]
        if max_minutes:
            train += ["--max_minutes", max_minutes]
        if (done or legacy) and not retrain:
            say(f"\n[O] 학습이 끝난 모델이 있음 ({CKPT}/best) → 학습 건너뜀. 처음부터 다시 하려면 --retrain", log)
        else:
            resume = has_save and not retrain
            say(f"\n=== 1. 정규화 모델 학습 ({'저장 지점부터 이어서' if resume else '처음부터'}, 5에폭 전체 약 40분~1시간 예상) ===",
                log)
            rc = run(train + (["--resume"] if resume else []), log)
            if rc not in (0, 130) and not resume and not any(
                    (ck / d / "train_state.pt").exists() for d in ("last", "last_old")):
                say("\n[!] 학습 실패. GPU 메모리 부족일 수 있어서 batch 8, grad_accum 2로 다시 시도", log)
                rc = run(train + ["--batch", "8", "--grad_accum", "2"], log)
            if rc not in (0, 130):
                say("\n[X] 학습 실패. results/week2_log.txt 를 보낼 것", log)
                return 1
            if not (ck / "train_done.json").exists():
                say("\n[멈춤] 학습이 아직 안 끝났음. 다시 실행하면 멈춘 곳부터 이어서 한다.", log)
                say("       (변환 점수와 A/B/C 번역은 학습이 끝난 실행에서 진행)", log)
                return 0

        # ---------------- 2. test 분할 변환 점수
        say("\n=== 2. test 분할 2000문장 변환 점수 ===", log)
        pred = [sys.executable, "scripts/predict_normalizer.py", "--ckpt", str(CKPT / "best"),
                "--input", "data/pairs/test.csv", "--n", "2000", "--out", "results/norm_test_pred.csv"]
        if run(pred, log) != 0:
            say("\n[X] 변환 실패. results/week2_log.txt 를 보낼 것", log)
            return 1

        # ---------------- 3. A/B/C 번역 확인
        say("\n=== 3. test 100쌍 A/B/C 중국어 번역 (MADLAD) ===", log)
        if run([sys.executable, "scripts/translate_pairs.py", "--ckpt", str(CKPT / "best")], log) != 0:
            say("\n[X] 번역 실패. results/week2_log.txt 를 보낼 것", log)
            return 1

        say(f"\n=== 완료 {datetime.now():%Y-%m-%d %H:%M} ===", log)
        say("  checkpoints/kobart_norm/train_log.csv : 에폭별 점수", log)
        say("  results/norm_test_pred.csv            : test 문장별 변환 결과", log)
        say("  results/데이터쌍_정리.xlsx            : 학습쌍 요약 + 샘플", log)
        say("  results/번역확인_test.xlsx            : A/B/C 번역 나란히", log)
        say("  results/week2_log.txt                 : 전체 출력", log)
        say("  데모: scripts\\run_demo.bat", log)
    return 0


if __name__ == "__main__":
    sys.exit(main())
