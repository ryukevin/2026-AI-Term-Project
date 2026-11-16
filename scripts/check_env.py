"""
새 PC에서 처음 실행해서 환경과 데이터 위치를 점검한다.

  python scripts/check_env.py

문제가 있으면 [X] 로 표시하고 해결 방법을 알려준다. 문제가 있으면 종료 코드 1.
"""
import importlib.util
import shutil
import sys
from pathlib import Path

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

ROOT = Path(__file__).resolve().parents[1]
bad = []


def ok(msg):
    print(f"[O] {msg}")


def ng(msg, fix):
    print(f"[X] {msg}\n    → {fix}")
    bad.append(msg)


def warn(msg):
    print(f"[!] {msg}")


print(f"저장소 위치: {ROOT}")
print(f"Python {sys.version.split()[0]} ({sys.executable})")

# ---------------- 패키지
need = ["torch", "transformers", "accelerate", "jiwer", "pandas", "soundfile", "librosa", "sacrebleu", "sentencepiece"]
missing = [m for m in need if importlib.util.find_spec(m) is None]
if missing:
    ng(f"패키지 없음: {', '.join(missing)}", "pip install -r requirements.txt")
else:
    ok("필수 패키지 모두 있음")

# ---------------- GPU
if importlib.util.find_spec("torch"):
    import torch
    if torch.cuda.is_available():
        p = torch.cuda.get_device_properties(0)
        vram = p.total_memory / 2**30
        ok(f"GPU {p.name}, VRAM {vram:.1f}GB (torch {torch.__version__})")
        if vram < 4:
            warn("VRAM 4GB 미만: Whisper turbo(fp16 약 2GB)는 되지만 번역 모델 비교는 어려움")
    else:
        if "+cpu" in torch.__version__:
            ng(f"torch가 CPU 버전 ({torch.__version__})",
               "pip uninstall torch 후 pip install torch --index-url https://download.pytorch.org/whl/cu128")
        else:
            ng("CUDA GPU를 못 찾음", "nvidia-smi 로 드라이버 확인")

# ---------------- 디스크 (Whisper 모델 약 1.6GB, 음성 zip 수십 GB)
free = shutil.disk_usage(ROOT).free / 2**30
if free > 50:
    ok(f"저장소 드라이브 여유 공간 {free:.0f}GB")
else:
    warn(f"저장소 드라이브 여유 공간 {free:.0f}GB (음성 zip이 크면 부족할 수 있음)")

# ---------------- 데이터
raw = ROOT / "data" / "raw"
for name, desc in (("dialect_gs", "014 방언 발화 라벨"), ("elder_gs", "139-1 중·노년층 라벨")):
    zips = list((raw / name).rglob("*.zip")) if (raw / name).exists() else []
    if zips:
        ok(f"{desc}: data/raw/{name} 에 zip {len(zips)}개")
    else:
        ng(f"{desc} 없음 (data/raw/{name})", "docs/음성작업_인수인계.md 의 '1. 데이터 넣기' 참고")

audio = raw / "audio"
wavs = list(audio.rglob("*.wav")) if audio.exists() else []
azips = list(audio.rglob("*.zip")) if audio.exists() else []
if wavs or azips:
    ok(f"음성: data/raw/audio 에 wav {len(wavs)}개, zip {len(azips)}개")
else:
    ng("음성 없음 (data/raw/audio)", "AI Hub 원천데이터(wav) zip을 data/raw/audio 에 넣는다 (풀지 않아도 됨)")

print()
if bad:
    print(f"문제 {len(bad)}개. 위 → 안내대로 고친 뒤 다시 실행.")
    sys.exit(1)
print("모두 정상. 다음: scripts\\run_whisper.bat")
