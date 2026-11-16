"""
AI Hub 방언 라벨 공용 로더.

두 데이터셋의 JSON 구조가 달라서 여기서 한 가지 형식으로 맞춘다.

1) 014. 한국어 방언 발화 데이터(경상도)  → format "dialect2020"
   - 대화 1개 = JSON 1개 (예: DKSR20001582.json), 음성은 같은 이름의 긴 wav
   - utterance[]: form, dialect_form, standard_form, speaker_id('1','2'), start, end(초 문자열)
   - speaker_id는 파일 안에서만 쓰는 번호다. 전역 화자 ID가 없어서
     화자 분리는 "대화 파일(=같은 화자 쌍)" 단위로 대신한다.

2) 139-1. 중·노년층 한국어 방언 데이터(경상도) → format "elder"
   - 녹음 1개 = JSON 1개 (예: talk_set2_..._11_0_76.json), 음성은 fileName.wav
   - transcription.sentences[]: speakerId(전역, 예 speakergs4909), dialect, standard,
     pronunciation, startTime/endTime("HH:MM:SS.mmm")
   - 종류: st_(1인 따라말하기, 대본 읽기) / say_(1인 질문에 답하기) / talk_(2인 대화)

라벨은 zip을 풀지 않고 바로 읽을 수 있다 (폴더, zip, json 모두 입력 가능).
"""
from __future__ import annotations

import json
import re
import zipfile
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Iterator


# ---------------------------------------------------------------- 레코드
@dataclass
class Utt:
    dataset: str      # dialect_gs / elder_gs
    subset: str       # conv / st / say / talk
    orig_split: str   # AI Hub가 나눠 둔 Training / Validation
    file_id: str      # JSON(=음성 파일) 이름, 확장자 없음
    group: str        # 학습/검증/시험 분리 기준 키
    speaker: str      # 화자 키
    dialect: str      # 방언 전사 (원본 그대로)
    standard: str     # 표준어 전사 (원본 그대로)
    form: str         # 원 전사 (014는 (방언)/(표준) 표기 포함)
    start: float
    end: float

    def to_dict(self):
        return asdict(self)


# ---------------------------------------------------------------- 파일 순회
def _load_bytes(b: bytes):
    for enc in ("utf-8-sig", "cp949"):
        try:
            return json.loads(b.decode(enc))
        except (UnicodeDecodeError, json.JSONDecodeError):
            continue
    return None


def iter_json(paths) -> Iterator[tuple[str, str, dict | None]]:
    """(원본경로, json이름, 파싱결과) 를 돌려준다. 파싱 실패하면 dict 대신 None."""
    if isinstance(paths, (str, Path)):
        paths = [paths]
    for p in map(Path, paths):
        if p.is_dir():
            files = sorted(list(p.rglob("*.zip")) + list(p.rglob("*.json")))
        else:
            files = [p]
        for f in files:
            if f.suffix.lower() == ".zip":
                with zipfile.ZipFile(f) as z:
                    for name in sorted(z.namelist()):
                        if name.lower().endswith(".json"):
                            yield str(f), name, _load_bytes(z.read(name))
            elif f.suffix.lower() == ".json":
                yield str(f), f.name, _load_bytes(f.read_bytes())


def detect_format(d: dict) -> str | None:
    if not isinstance(d, dict):
        return None
    if isinstance(d.get("utterance"), list):
        return "dialect2020"
    if isinstance(d.get("transcription"), dict):
        return "elder"
    return None


def _orig_split(src: str) -> str:
    s = src.lower()
    if "valid" in s or "vl_" in s or "_2.zip" in s:
        return "Validation"
    return "Training"


def hms_to_sec(t) -> float:
    """'00:01:15.140' 또는 '75.14' → 초"""
    t = str(t).strip()
    if ":" in t:
        h, m, s = t.split(":")
        return int(h) * 3600 + int(m) * 60 + float(s)
    return float(t)


def _s(v) -> str:
    v = "" if v is None else str(v)
    return "" if v.strip() in ("None", "null", "nan") else v


def parse(src: str, name: str, d: dict) -> list[Utt]:
    fmt = detect_format(d)
    split = _orig_split(src)
    stem = Path(name).stem
    out: list[Utt] = []
    if fmt == "dialect2020":
        fid = str(d.get("id") or stem)
        for u in d["utterance"]:
            spk = str(u.get("speaker_id", ""))
            try:
                st, en = float(u.get("start")), float(u.get("end"))
            except (TypeError, ValueError):
                st = en = float("nan")
            out.append(Utt("dialect_gs", "conv", split, fid, group=f"dialect_gs:{fid}",
                           speaker=f"{fid}#{spk}",
                           dialect=_s(u.get("dialect_form")), standard=_s(u.get("standard_form")),
                           form=_s(u.get("form")), start=st, end=en))
    elif fmt == "elder":
        fid = str(d.get("fileName") or stem)
        subset = fid.split("_")[0]  # st / say / talk
        for s in d["transcription"].get("sentences") or []:
            spk = str(s.get("speakerId", ""))
            try:
                st, en = hms_to_sec(s.get("startTime")), hms_to_sec(s.get("endTime"))
            except (TypeError, ValueError):
                st = en = float("nan")
            out.append(Utt("elder_gs", subset, split, fid, group=f"elder_gs:{spk}",
                           speaker=spk,
                           dialect=_s(s.get("dialect")), standard=_s(s.get("standard")),
                           form=_s(s.get("pronunciation")), start=st, end=en))
    return out


# ---------------------------------------------------------------- 텍스트 정리
NAME_MASK = re.compile(r"&[^&\s]*&|@\S*이름\S*|#[^#\s]*#")
UNK = re.compile(r"\(\(\s*\)\)|\(\(x+\)\)", re.I)       # 알아들을 수 없음
DUAL = re.compile(r"\(([^()/]*)\)/\(([^()/]*)\)")        # (방언)/(표준) 이중 전사


def has_mask(text: str) -> bool:
    """개인정보 마스킹(&name1&, @이름1, #이름#)이나 '알아들을 수 없음(())'이 있으면 True"""
    return bool(NAME_MASK.search(text) or UNK.search(text))


def clean_text(text: str, side: str = "dialect") -> str:
    """
    전사 기호만 지우고 방언 어휘·종결 어미는 그대로 둔다.
      {laughing} {clearing} 등 소리 태그  → 삭제
      ((단어))  불확실 전사                → 단어만 남김
      (방언)/(표준) 이중 전사              → side에 맞는 쪽 선택
      -그- 같은 더듬음/반복 표시            → 삭제
      음~ 어~ 의 '~'                       → '~'만 삭제
    """
    t = str(text or "")
    t = re.sub(r"\{[^}]*\}", " ", t)
    t = DUAL.sub(lambda m: m.group(1) if side == "dialect" else m.group(2), t)
    t = re.sub(r"\(\(([^)]*)\)\)", r"\1", t)
    t = re.sub(r"(?<!\S)-[^\s-]+-(?!\S)", " ", t)
    t = t.replace("~", "")
    t = re.sub(r"[*#]", "", t)
    return re.sub(r"\s+", " ", t).strip()


def norm_key(text: str) -> str:
    """비교용 키: 공백·문장부호 제거"""
    return re.sub(r"[^\w]", "", str(text or ""))
