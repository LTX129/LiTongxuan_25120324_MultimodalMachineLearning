import re

# 一些常见参考文献特征
BRACKET_CIT = re.compile(r"\[\s*\d{1,4}\s*\]")          # [12]
YEAR_CIT = re.compile(r"\(\s*(19|20)\d{2}\s*\)")        # (2023)
ARXIV = re.compile(r"\barxiv\b|arXiv:\s*\d{4}\.\d{4,5}", re.IGNORECASE)
DOI = re.compile(r"\bdoi\b\s*[:/]", re.IGNORECASE)
PP = re.compile(r"\bpp\.\s*\d", re.IGNORECASE)
ETAL = re.compile(r"\bet\s+al\.\b", re.IGNORECASE)
PROC = re.compile(r"\bReferences\b|\bproceedings of\b|\bconference\b|\bjournal\b|\bICML\b|\bNeurIPS\b|\bCVPR\b", re.IGNORECASE)

def is_reference_like(text: str) -> bool:
    """
    Heuristic detector for reference/bibliography chunks.
    Designed to be robust even if the PDF doesn't have a clean 'References' header.
    """
    if not text:
        return False

    t = text.strip()
    if len(t) < 80:
        return False

    lower = t.lower()

    # 强特征：明显的 references 标题
    if lower.startswith("references") or lower.startswith("bibliography"):
        return True

    # 统计引用特征命中数
    hits = 0
    hits += len(BRACKET_CIT.findall(t))
    hits += len(YEAR_CIT.findall(t))
    if ARXIV.search(t): hits += 2
    if DOI.search(t): hits += 2
    if PP.search(t): hits += 1
    if ETAL.search(t): hits += 1
    if PROC.search(t): hits += 1

    # 数字和标点比例（参考文献通常更高）
    digits = sum(ch.isdigit() for ch in t)
    punct = sum(ch in ".,;:()[]{}-/“”\"" for ch in t)
    ratio = (digits + punct) / max(len(t), 1)

    # 判定阈值：满足其一即可
    if hits >= 6:
        return True
    if hits >= 3 and ratio > 0.22:
        return True
    if ratio > 0.30 and hits >= 2:
        return True

    return False