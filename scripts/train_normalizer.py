"""
사투리 → 표준어 정규화 모델 학습 (KoBART 또는 한국어 T5 계열 파인튜닝)

  # 1) 파이프라인이 끝까지 도는지 먼저 확인 (2000쌍, 1에폭, 몇 분)
  python scripts/train_normalizer.py --limit 2000 --epochs 1 --out checkpoints/smoke

  # 2) 본 학습 (train_50k.csv)
  python scripts/train_normalizer.py

  # 3) 데이터를 늘릴 때
  python scripts/train_normalizer.py --train data/pairs/train.csv --out checkpoints/kobart_norm_full

입력 CSV 열: dialect, standard (build_pairs.py 출력 그대로)

하는 일
  - 매 에폭(또는 --eval_every 스텝)마다 검증 손실과 생성 품질(chrF, 글자 BLEU, CER)을 잰다.
  - 생성 품질 옆에 "입력을 그대로 복사했을 때" 점수를 같이 찍는다. 모델 점수가 이보다 높아야 의미가 있다.
  - 검증 chrF가 가장 높은 시점을 best/ 에 저장하고, --patience 번 연속 안 오르면 멈춘다 (early stopping).

출력 (checkpoints/<이름>/)
  best/                 모델 가중치 + 토크나이저 + norm_config.json (추론에 필요한 설정)
  train_log.csv         평가 시점별 손실·점수
  valid_samples_best.csv  best 시점의 검증 문장별 입력/정답/출력/chrF (오류 분석용)

checkpoints/ 는 .gitignore에 있어서 GitHub에 올라가지 않는다. 팀 공유는 드라이브로.
"""
import argparse
import json
import math
import random
import sys
import time
from datetime import datetime
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.aihub import norm_key  # noqa: E402
from src.normalize import CONFIG_NAME, NORM_DECODE, default_prefix, encode_batch, generate, pad_batch  # noqa: E402

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")


def load_pairs(path, limit=0, seed=42):
    df = pd.read_csv(path).fillna("")
    for col in ("dialect", "standard"):
        if col not in df.columns:
            sys.exit(f"{path}에 '{col}' 열이 없음")
    df = df[(df.dialect.astype(str).str.strip() != "") & (df.standard.astype(str).str.strip() != "")]
    if limit and len(df) > limit:
        df = df.sample(n=limit, random_state=seed)
    return df.reset_index(drop=True)


def gen_scores(pred, ref):
    import jiwer
    import sacrebleu
    p = [norm_key(x) for x in pred]
    r = [norm_key(x) for x in ref]
    return {
        "chrF": round(sacrebleu.corpus_chrf(list(pred), [list(ref)]).score, 2),
        "BLEU_char": round(sacrebleu.corpus_bleu(list(pred), [list(ref)], tokenize="char").score, 2),
        "CER": round(jiwer.cer(r, p), 4),
        "exact": round(sum(a == b for a, b in zip(p, r)) / max(len(r), 1), 4),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train", default="data/pairs/train_50k.csv")
    ap.add_argument("--valid", default="data/pairs/valid.csv")
    ap.add_argument("--model", default="gogamza/kobart-base-v2",
                    help="사전학습 모델 (예: gogamza/kobart-base-v2, paust/pko-t5-base)")
    ap.add_argument("--out", default="checkpoints/kobart_norm")
    ap.add_argument("--prefix", default=None, help="입력 앞 접두어. 생략하면 T5만 '방언을 표준어로 바꾸기: '")
    ap.add_argument("--max_len", type=int, default=128, help="입력·정답 최대 토큰 수")
    ap.add_argument("--epochs", type=int, default=5)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--grad_accum", type=int, default=1, help="메모리 부족하면 batch를 줄이고 이걸 늘린다")
    ap.add_argument("--lr", type=float, default=5e-5)
    ap.add_argument("--warmup", type=float, default=0.05, help="전체 스텝 중 워밍업 비율")
    ap.add_argument("--weight_decay", type=float, default=0.01)
    ap.add_argument("--eval_every", type=int, default=0, help="N 스텝마다 평가 (0이면 에폭마다)")
    ap.add_argument("--n_valid_loss", type=int, default=5000, help="검증 손실에 쓸 문장 수")
    ap.add_argument("--n_valid_gen", type=int, default=500, help="생성 품질 평가에 쓸 문장 수 (생성은 느림)")
    ap.add_argument("--patience", type=int, default=2, help="best가 이만큼 연속 안 오르면 중단")
    ap.add_argument("--precision", default="auto", choices=["auto", "bf16", "fp16", "fp32"])
    ap.add_argument("--limit", type=int, default=0, help="빠른 시험용: train을 이 개수만 사용")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    import torch
    from torch.utils.data import DataLoader
    from transformers import AutoModelForSeq2SeqLM, AutoTokenizer, get_linear_schedule_with_warmup

    random.seed(args.seed)
    torch.manual_seed(args.seed)
    out = Path(args.out)
    (out / "best").mkdir(parents=True, exist_ok=True)

    # ---------------- 장치·정밀도
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    prec = args.precision
    if prec == "auto":
        prec = "fp32" if dev == "cpu" else ("bf16" if torch.cuda.is_bf16_supported() else "fp16")
    amp_dtype = {"bf16": torch.bfloat16, "fp16": torch.float16}.get(prec)
    scaler = torch.amp.GradScaler("cuda") if prec == "fp16" else None
    gpu = torch.cuda.get_device_name(0) if dev == "cuda" else "CPU"
    print(f"장치 {gpu}, 정밀도 {prec}")
    if dev == "cpu":
        print("[!] GPU가 없음. CPU 학습은 매우 느리다. torch가 CPU 버전으로 깔렸는지 확인 (scripts/check_env.py)")

    # ---------------- 데이터
    tr = load_pairs(args.train, args.limit, args.seed)
    va = load_pairs(args.valid)
    va_loss = va.sample(n=min(args.n_valid_loss, len(va)), random_state=args.seed).reset_index(drop=True)
    va_gen = va.sample(n=min(args.n_valid_gen, len(va)), random_state=args.seed + 1).reset_index(drop=True)
    print(f"train {len(tr)}쌍 ({args.train}), valid 손실용 {len(va_loss)} / 생성용 {len(va_gen)} ({args.valid})")

    # ---------------- 모델
    prefix = default_prefix(args.model) if args.prefix is None else args.prefix
    tok = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForSeq2SeqLM.from_pretrained(args.model).to(dev)
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token
    if getattr(model.config, "decoder_start_token_id", None) is None:
        model.config.decoder_start_token_id = tok.eos_token_id
    pad = tok.pad_token_id
    n_param = sum(p.numel() for p in model.parameters()) / 1e6
    print(f"모델 {args.model} ({n_param:.0f}M 파라미터), 접두어 {prefix!r}")

    # ---------------- 토큰화 (한 번만)
    def tokenize(df):
        x = encode_batch(tok, df.dialect.tolist(), args.max_len, prefix)
        y = encode_batch(tok, df.standard.tolist(), args.max_len)
        return list(zip(x, y))

    t0 = time.time()
    tr_tok, va_tok = tokenize(tr), tokenize(va_loss)
    lens = pd.Series([len(x) for x, _ in tr_tok] + [len(y) for _, y in tr_tok])
    cut = sum(len(x) >= args.max_len for x, _ in tr_tok) + sum(len(y) >= args.max_len for _, y in tr_tok)
    print(f"토큰화 {time.time() - t0:.0f}초. 토큰 길이 중앙값 {lens.median():.0f}, 99% {lens.quantile(.99):.0f}, "
          f"최대 {lens.max()} → max_len {args.max_len}에서 잘린 문장 {cut}개")
    if cut > 0.01 * len(lens):
        print(f"[!] 1% 넘게 잘림. --max_len을 {int(lens.quantile(.995)) + 8} 정도로 올리는 걸 고려")

    def collate(batch):
        x, xm = pad_batch([b[0] for b in batch], pad)
        y, _ = pad_batch([b[1] for b in batch], pad, label=True)
        return {"input_ids": x, "attention_mask": xm, "labels": y}

    # Windows에서 멀티프로세스 로더가 문제를 일으키므로 num_workers=0
    tr_dl = DataLoader(tr_tok, batch_size=args.batch, shuffle=True, collate_fn=collate, num_workers=0)
    va_dl = DataLoader(va_tok, batch_size=args.batch * 2, shuffle=False, collate_fn=collate, num_workers=0)

    steps_per_epoch = math.ceil(len(tr_dl) / args.grad_accum)
    total = steps_per_epoch * args.epochs
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    sched = get_linear_schedule_with_warmup(opt, int(total * args.warmup), total)
    eval_every = args.eval_every or steps_per_epoch
    print(f"에폭당 {steps_per_epoch} 스텝, 총 {total} 스텝, {eval_every} 스텝마다 평가\n")

    def autocast():
        return torch.autocast("cuda", dtype=amp_dtype) if amp_dtype is not None else torch.autocast("cpu", enabled=False)

    # ---------------- 복사 기준선 (모델 없이 입력을 그대로 낸 경우)
    copy = gen_scores(va_gen.dialect.tolist(), va_gen.standard.tolist())
    print(f"복사 기준선 (입력 그대로): chrF {copy['chrF']}, 글자BLEU {copy['BLEU_char']}, CER {copy['CER']}\n")

    def evaluate():
        model.eval()
        tot, n = 0.0, 0
        with torch.inference_mode():
            for b in va_dl:
                b = {k: v.to(dev) for k, v in b.items()}
                with autocast():
                    loss = model(**b).loss
                k = (b["labels"] != -100).sum().item()
                tot += loss.float().item() * k
                n += k
        preds = generate(model, tok, va_gen.dialect.tolist(), args.max_len, prefix, batch_size=args.batch * 2)
        model.train()
        return tot / max(n, 1), preds

    # ---------------- 학습
    log_rows, best, bad, step, ep = [], -1.0, 0, 0, 0
    run_loss, run_n, stop = 0.0, 0, False
    if dev == "cuda":
        torch.cuda.reset_peak_memory_stats()
    t_start = time.time()
    model.train()
    for ep in range(1, args.epochs + 1):
        for i, b in enumerate(tr_dl, 1):
            b = {k: v.to(dev) for k, v in b.items()}
            with autocast():
                loss = model(**b).loss / args.grad_accum
            if scaler:
                scaler.scale(loss).backward()
            else:
                loss.backward()
            run_loss += loss.item() * args.grad_accum
            run_n += 1
            if i % args.grad_accum and i != len(tr_dl):
                continue
            if scaler:
                scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            if scaler:
                scaler.step(opt)
                scaler.update()
            else:
                opt.step()
            sched.step()
            opt.zero_grad(set_to_none=True)
            step += 1

            if step % 50 == 0:
                el = time.time() - t_start
                print(f"  에폭 {ep} 스텝 {step}/{total}  train_loss {run_loss / run_n:.4f}  "
                      f"lr {sched.get_last_lr()[0]:.2e}  경과 {el / 60:.1f}분 (남은 예상 {el / step * (total - step) / 60:.1f}분)",
                      flush=True)

            if step % eval_every == 0 or step == total:
                v_loss, preds = evaluate()
                sc = gen_scores(preds, va_gen.standard.tolist())
                row = {"epoch": ep, "step": step, "train_loss": round(run_loss / max(run_n, 1), 4),
                       "valid_loss": round(v_loss, 4), **sc, "copy_chrF": copy["chrF"],
                       "minutes": round((time.time() - t_start) / 60, 1)}
                log_rows.append(row)
                run_loss, run_n = 0.0, 0
                pd.DataFrame(log_rows).to_csv(out / "train_log.csv", index=False, encoding="utf-8-sig")
                mark = ""
                if sc["chrF"] > best:
                    best, bad, mark = sc["chrF"], 0, "  ← best 저장"
                    model.save_pretrained(out / "best")
                    tok.save_pretrained(out / "best")
                    cfg = {"base_model": args.model, "prefix": prefix, "max_len": args.max_len,
                           "decode": NORM_DECODE, "train_file": args.train, "n_train": len(tr),
                           "epoch": ep, "step": step, "valid": {**sc, "valid_loss": round(v_loss, 4)},
                           "copy_baseline": copy, "saved_at": datetime.now().isoformat(timespec="seconds")}
                    (out / "best" / CONFIG_NAME).write_text(json.dumps(cfg, ensure_ascii=False, indent=2),
                                                            encoding="utf-8")
                    import sacrebleu
                    pd.DataFrame({"dialect": va_gen.dialect, "standard": va_gen.standard, "pred": preds,
                                  "chrF": [round(sacrebleu.sentence_chrf(p, [r]).score, 1)
                                           for p, r in zip(preds, va_gen.standard)]}) \
                        .to_csv(out / "valid_samples_best.csv", index=False, encoding="utf-8-sig")
                else:
                    bad += 1
                print(f"[평가] 에폭 {ep} 스텝 {step}: valid_loss {v_loss:.4f}  chrF {sc['chrF']} "
                      f"(복사 {copy['chrF']})  글자BLEU {sc['BLEU_char']}  CER {sc['CER']}  완전일치 {sc['exact']}{mark}")
                for d, s, p in list(zip(va_gen.dialect, va_gen.standard, preds))[:3]:
                    print(f"    사투리 {d}\n    정답   {s}\n    출력   {p}")
                if bad >= args.patience:
                    print(f"\n{args.patience}번 연속 개선 없음 → 조기 종료")
                    stop = True
                    break
        if stop:
            break

    mem = f", 최대 VRAM {torch.cuda.max_memory_allocated() / 1e9:.2f}GB" if dev == "cuda" else ""
    print(f"\n완료: {(time.time() - t_start) / 60:.1f}분{mem}")
    print(f"best 검증 chrF {best} (복사 기준선 {copy['chrF']}) → {out / 'best'}")
    if best <= copy["chrF"]:
        print("[!] 모델이 복사 기준선을 못 넘었음. 학습률·에폭·데이터를 점검할 것")


if __name__ == "__main__":
    main()
