# 2026 인공지능 Term Project (13조)

영남대학교 로봇공학과 「인공지능」(양준모 교수) 팀 프로젝트

**주제: 경상도 사투리 → 표준어 → 외국어(중국어) 번역**
표준어를 거치면 번역이 얼마나 좋아지는지를 A(사투리 그대로) / B(우리 모델 표준어) / C(정답 표준어) 세 조건으로 비교한다.

## 팀원
| 이름 | GitHub | 역할 |
|---|---|---|
| 류승익 | [@ryukevin](https://github.com/ryukevin) | 모델 설계·학습, 코드 통합 |
| 이효민 | [@hyoyu0309](https://github.com/hyoyu0309) | 데이터 수집·전처리 |
| 주상우 | [@zhou0729](https://github.com/zhou0729)| 베이스라인·시연 프로토타입 |
| 차희태 | [@ckgmlxo2002](https://github.com/ckgmlxo2002) | 실험 분석·자료 조사 |

## 폴더
```
data/            AI Hub 원본·가공본 (git 제외, 재배포 금지)
  raw/dialect_gs/   014. 한국어 방언 발화(경상도) 라벨 zip
  raw/elder_gs/     139-1. 중·노년층 방언(경상도) 라벨 zip
  raw/audio/        원천 음성 wav (Whisper CER 측정용, 아직 없음)
  pairs/            train/valid/test CSV (build_pairs.py 출력)
  eval/             평가 세트 100문장 (eval_set.csv), 번역 모델 비교 문장 (mt_pilot.csv)
scripts/         실행 스크립트
src/             공용 모듈 (aihub.py 라벨 로더, translate.py 번역기)
results/         점수표 CSV
```

## 설치
```bash
pip install torch --index-url https://download.pytorch.org/whl/cu128
pip install -r requirements.txt
```

## 실행 순서
모든 명령은 저장소 최상위 폴더에서 실행한다. 경로는 전부 상대 경로.

### 0. 데이터 넣기
AI Hub에서 받은 **라벨 zip을 풀지 않고** 그대로 넣는다.
```
data/raw/dialect_gs/Training/(비식별화완료)경상도_학습데이터_1.zip
data/raw/dialect_gs/Validation/(비식별화완료)경상도_학습데이터_2.zip
data/raw/elder_gs/Training/TL_02. 경상도_0X. ....zip     (강원도 TL_01은 넣지 않음)
data/raw/elder_gs/Validation/VL_02. 경상도_0X. ....zip
```

### 1. 데이터 구조 확인
```bash
python scripts/whisper_cer_aihub.py --label_dir data/raw/dialect_gs data/raw/elder_gs --inspect --max_files 1000
```

### 2. 학습쌍 만들기
```bash
python scripts/build_pairs.py
```
→ `data/pairs/train.csv, valid.csv, test.csv, train_50k.csv, build_log.txt(필터 단계별 개수)`
평가 세트(`data/eval/*.csv`)가 생기면 **다시 실행**해서 평가 문장을 학습 데이터에서 빼야 한다.

### 3. Whisper CER (음성 wav 필요)
```bash
python scripts/whisper_cer_aihub.py --label_dir data/raw/dialect_gs --audio_dir data/raw/audio --n 30 --tag dialect
```
→ `results/whisper_cer_dialect.csv`, `results/whisper_cer_summary.csv`

### 4. 번역 모델 고르기 (NLLB 1.3B vs MADLAD 3B)
```bash
python scripts/compare_mt.py --from_pairs data/pairs/test.csv --n 25
```
→ `data/eval/mt_pilot.csv`의 zh 칸을 원어민이 채운 뒤 `python scripts/compare_mt.py` 로 다시 실행하면 chrF까지 나온다.
한 번 고른 모델과 디코딩 설정(`src/translate.py`의 `DECODE`)은 끝까지 바꾸지 않는다.

### 5. A/B/C 평가
`data/eval/eval_set.csv` 열: `id, dialect, standard, zh` (+ B 조건용 `pred_standard`)
```bash
python scripts/evaluate.py --eval data/eval/eval_set.csv --translate --model nllb --tag nllb
python scripts/evaluate.py --eval data/eval/eval_set.csv --tag nllb --human_sheet   # 원어민 블라인드 평가지
```
→ `results/scores_nllb.csv`(조건별 BLEU·chrF·95% 구간), `scores_diff_nllb.csv`, `norm_scores_nllb.csv`, `eval_detail_nllb.csv`
