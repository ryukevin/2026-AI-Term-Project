"""
13조 사투리 번역 데모 (Gradio)

  scripts\\run_demo.bat        (더블클릭)
  python app.py               → 브라우저에서 http://127.0.0.1:7860
  python app.py --share       → 다른 사람도 들어올 수 있는 임시 링크 (72시간)

사투리 → 표준어(우리 KoBART) → 외국어(MADLAD) 를 한 화면에 보여준다.
"직접 번역"은 표준어를 거치지 않고 사투리를 그대로 번역한 결과(A 조건)라서, 정규화 효과를 바로 비교할 수 있다.
여러 줄을 넣으면 줄마다 따로 처리한다.

음성 입력(Whisper)은 CER 결과를 보고 넣을지 정한다. 넣게 되면 이 화면 위쪽에 음성 업로드 칸과
"전사문 수정" 칸을 붙이고, 수정된 텍스트를 아래 run()에 넘기면 된다.
"""
import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

LANG_LABEL = {"zh": "중국어 (간체)", "en": "영어", "ja": "일본어", "vi": "베트남어",
              "th": "태국어", "ru": "러시아어", "es": "스페인어"}
CKPT_CANDIDATES = ["checkpoints/kobart_norm/best", "checkpoints/smoke/best"]
EXAMPLES = [
    ["니 밥 뭇나?"],
    ["어데 가노? 퍼뜩 온나."],
    ["와 이리 덥노, 쪼매만 쉬었다 가자."],
    ["단디 해라, 알았제?"],
    ["내 억수로 피곤하다 아이가."],
]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default="", help="정규화 모델 (생략하면 kobart_norm → smoke 순서로 찾음)")
    ap.add_argument("--mt", default="madlad", help="번역 모델 (평가와 같은 것)")
    ap.add_argument("--share", action="store_true")
    ap.add_argument("--port", type=int, default=7860)
    args = ap.parse_args()

    import gradio as gr
    from src.pipeline import Pipeline

    ckpt = args.ckpt or next((c for c in CKPT_CANDIDATES if (ROOT / c / "config.json").exists()), "")
    if not ckpt:
        sys.exit("[X] 정규화 모델이 없음 → scripts\\run_week2.bat 으로 먼저 학습")
    print(f"모델 불러오는 중... 정규화 {ckpt}, 번역 {args.mt} (처음엔 1~2분)")
    pipe = Pipeline(ckpt, args.mt)
    print("준비 완료")

    def run(text, tgt, compare):
        lines = [s.strip() for s in str(text or "").splitlines() if s.strip()]
        if not lines:
            return "", "", "", "사투리 문장을 입력하세요."
        rows = pipe.run(lines, tgt, compare)
        std = "\n".join(r["standard"] for r in rows)
        trans = "\n".join(r["translation"] for r in rows)
        direct = "\n".join(r.get("direct_translation", "") for r in rows) if compare else ""
        t_norm = sum(r["sec_norm"] for r in rows)
        t_mt = sum(r["sec_mt"] for r in rows)
        info = f"처리 시간: 표준어 변환 {t_norm:.2f}초 · 번역 {t_mt:.2f}초 ({len(rows)}문장)"
        return std, trans, direct, info

    with gr.Blocks(title="13조 사투리 번역기") as demo:
        gr.Markdown("# 경상도 사투리 → 표준어 → 외국어 번역\n"
                    "사투리를 표준어로 먼저 바꾼 다음 번역합니다. 여러 줄을 넣으면 줄마다 따로 처리합니다.")
        with gr.Row():
            with gr.Column():
                inp = gr.Textbox(label="사투리 입력", lines=4, placeholder="예: 니 밥 뭇나?")
                with gr.Row():
                    tgt = gr.Dropdown(choices=[(v, k) for k, v in LANG_LABEL.items()], value="zh", label="번역할 언어")
                    compare = gr.Checkbox(value=True, label="표준어를 거치지 않은 직접 번역도 보기")
                btn = gr.Button("번역", variant="primary")
                gr.Examples(EXAMPLES, inputs=[inp])
            with gr.Column():
                out_std = gr.Textbox(label="① 표준어 변환 (우리 모델)", lines=4)
                out_tr = gr.Textbox(label="② 표준어 → 번역", lines=4)
                out_direct = gr.Textbox(label="비교: 사투리를 그대로 번역", lines=4)
                info = gr.Markdown()
        btn.click(run, inputs=[inp, tgt, compare], outputs=[out_std, out_tr, out_direct, info])
        inp.submit(run, inputs=[inp, tgt, compare], outputs=[out_std, out_tr, out_direct, info])

    demo.launch(share=args.share, server_port=args.port, inbrowser=True)


if __name__ == "__main__":
    main()
