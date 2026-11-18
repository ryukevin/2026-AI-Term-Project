"""
번역 모델 비교(NLLB 1.3B vs MADLAD 3B)를 한 번에 돌린다. Kevin PC(RTX 3060 Ti)용.

  scripts\\run_compare_mt.bat      (더블클릭)
  python scripts/run_compare_mt.py

하는 일
  1) CUDA GPU가 잡히는지 확인 (없으면 중단)
  2) data/pairs/test.csv 가 있는지 확인
  3) compare_mt.py --from_pairs data/pairs/test.csv --n 25 실행
  4) 실패하면 (대개 GPU 메모리 부족) --batch_size 1 로 한 번 더

출력: data/eval/mt_pilot.csv, results/mt_compare.csv, results/mt_compare_summary.csv
"""
import subprocess
import sys
from pathlib import Path

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

ROOT = Path(__file__).resolve().parents[1]
PAIRS = Path("data/pairs/test.csv")


def main() -> int:
    try:
        import torch
    except ImportError:
        print("[X] torch가 없음 → README의 '설치' 순서대로 설치")
        return 1
    if not torch.cuda.is_available():
        print(f"[X] CUDA GPU를 못 찾음 (torch {torch.__version__}). 3060 Ti PC에서 실행할 것.")
        return 1
    p = torch.cuda.get_device_properties(0)
    print(f"[O] GPU {p.name}, VRAM {p.total_memory / 2**30:.1f}GB")

    if not (ROOT / PAIRS).exists():
        print(f"[X] {PAIRS} 없음 → python scripts/build_pairs.py 먼저 실행")
        return 1
    print(f"[O] {PAIRS} 있음")

    cmd = [sys.executable, "scripts/compare_mt.py", "--from_pairs", str(PAIRS), "--n", "25"]
    for bs in ("4", "1"):
        print(f"\n=== compare_mt.py 실행 (batch_size {bs}) ===", flush=True)
        if subprocess.run(cmd + ["--batch_size", bs], cwd=ROOT).returncode == 0:
            print("\n완료.")
            print("  results/mt_compare_summary.csv : 모델별 속도, VRAM")
            print("  results/mt_compare.csv         : 문장별 두 모델 번역 나란히")
            print("  data/eval/mt_pilot.csv         : 이 파일을 주상우에게 보내 zh 칸 채우기")
            print("  zh가 채워지면: python scripts/compare_mt.py  (chrF까지 계산)")
            return 0
        if bs == "4":
            print("\n[!] 실패. GPU 메모리 부족일 수 있어서 batch_size 1로 다시 시도", flush=True)
    print("\n[X] 두 번 다 실패. 위 에러 메시지를 복사해서 보낼 것")
    return 1


if __name__ == "__main__":
    sys.exit(main())
