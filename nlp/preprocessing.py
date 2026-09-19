import re

PHONE_RE = re.compile(r"\b(?:01[016789])[-.\s]?\d{3,4}[-.\s]?\d{4}\b")
EMAIL_RE = re.compile(r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b")
RRN_RE = re.compile(r"\b\d{6}[-\s]?[1-4]\d{6}\b")

def mask_pii(text: str) -> str:
    text = str(text)
    text = PHONE_RE.sub("[전화번호]", text)
    text = EMAIL_RE.sub("[이메일]", text)
    text = RRN_RE.sub("[주민번호]", text)
    return text

def clean_text(text: str) -> str:
    text = mask_pii(text)
    text = re.sub(r"\s+", " ", text).strip()
    return text
