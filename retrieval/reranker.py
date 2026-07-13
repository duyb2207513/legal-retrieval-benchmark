"""Rerank candidate sau RRF — sửa Contextual Relevancy thấp: RRF chỉ xếp
theo rank giữa 2 nguồn (vector/BM25), không đánh giá lại độ liên quan ngữ
nghĩa thật với câu hỏi gốc, nên top-MAX_COMPONENTS sau RRF có thể lẫn
candidate lạc đề hoàn toàn.

=== BUG Ở BẢN CŨ (đã fix) ===
Bản trước dùng rank_bm25.BM25Okapi coi CHÍNH TẬP SEEDS hiện tại làm mini-
corpus, rồi chuẩn hoá điểm bằng scale = 10 / max(raw_scores). Vấn đề: phép
chuẩn hoá này luôn kéo candidate tốt NHẤT TRONG TẬP lên ~10/10, bất kể nó
có thật sự liên quan tới câu hỏi hay không — nên rerank KHÔNG BAO GIỜ có
thể loại bỏ toàn bộ 1 tập seeds sai domain (vd cả 15-25 candidate sau RRF
đều lạc đề), nó chỉ sắp xếp lại "con nào đỡ tệ hơn trong đàn dê lạc". Đây
chính là lý do contextual_relevancy vẫn ~0 dù đã bật use_rerank.

=== CÁCH FIX ===
Thay self-referential normalization bằng một chỉ số TUYỆT ĐỐI, không phụ
thuộc vào các candidate khác trong tập: coverage = tỉ lệ từ khoá pháp lý
riêng biệt của câu hỏi thực sự xuất hiện trong văn bản candidate đó.
- coverage = |query_terms ∩ doc_terms| / |query_terms|
- Nếu 1 candidate không chứa TỪ KHOÁ NÀO của câu hỏi → coverage = 0, bị
  loại thẳng bất kể các candidate khác trong tập tệ tới đâu.
- Stopword list lọc bớt hư từ tiếng Việt phổ biến (và, của, là, cho...)
  để coverage không bị "lạm phát" giả tạo bởi các từ không mang nghĩa.
- Raw BM25 score (rank_bm25, vẫn coi seeds là mini-corpus) chỉ dùng làm
  tie-break PHỤ giữa các candidate đã cùng vượt ngưỡng coverage tuyệt đối
  — không dùng để quyết định pass/fail nữa.

Cách dùng (giữ nguyên signature, không cần đổi gì ở pipeline.py/config.py):
    from retrieval.reranker import rerank
    reranked = rerank(question, seeds, top_n=5, min_score=4.0)

min_score vẫn theo thang 0-10 như cũ (khớp RERANK_MIN_SCORE=4.0 trong
config.py), nhưng giờ 4.0 nghĩa là "candidate chứa >=40% từ khoá pháp lý
riêng biệt của câu hỏi" — một ngưỡng có ý nghĩa tuyệt đối, không phải
ngưỡng tương đối trong nội bộ tập seeds.
"""
from __future__ import annotations

import logging
import re

from config import MAX_CHARS_PER_UNIT

logger = logging.getLogger(__name__)

# Tokenizer đơn giản: chữ + số (kể cả có dấu tiếng Việt), lowercase.
_TOKEN_RE = re.compile(r"\w+", re.UNICODE)

# Hư từ tiếng Việt phổ biến — loại khỏi query_terms trước khi tính coverage,
# tránh coverage bị lạm phát giả tạo (candidate nào cũng có "và", "của"...).
# Không cần đầy đủ tuyệt đối — chỉ cần loại các từ tần suất cao nhất, vì
# coverage chỉ quan tâm tới TỪ KHOÁ MANG NGHĨA của câu hỏi.
_STOPWORDS = frozenset("""
và của là cho có không được về theo này đó khi nào như vậy các một những
tôi bạn em anh chị mình ạ nhé nếu thì ở trong ngoài từ đến với hay hoặc
đã sẽ đang bị phải cần nên do vì nên_là_gì gì ai đâu bao_nhiêu cách
""".split())


def _tokenize(text: str) -> list[str]:
    return _TOKEN_RE.findall((text or "").lower())


def _query_terms(question: str) -> set[str]:
    """Tập từ khoá riêng biệt của câu hỏi, đã loại hư từ — dùng làm mẫu số
    TUYỆT ĐỐI cho coverage (không phụ thuộc tập candidate)."""
    tokens = _tokenize(question)
    return {t for t in tokens if t not in _STOPWORDS and len(t) > 1}


def _build_doc_text(row: dict, max_chars: int) -> str:
    text = (row.get("text") or "")[:max_chars]
    return f"{row.get('citation', '')} {row.get('title_text') or ''} {text}"

import math

def rerank(
    question: str,
    seeds: list[dict],
    top_n: int = 5,
    min_score: float = 4.0,
    llm=None,
) -> list[dict]:
    if not seeds:
        return seeds

    query_terms = _query_terms(question)
    if not query_terms:
        logger.warning("rerank: câu hỏi không trích được từ khoá nào sau khi lọc stopword, fallback về seeds gốc.")
        return seeds[:top_n]

    # --- Build doc_terms cho từng seed trước, dùng để tính IDF ---
    doc_terms_list = [set(_tokenize(_build_doc_text(row, MAX_CHARS_PER_UNIT))) for row in seeds]
    n_docs = len(doc_terms_list)

    # IDF cho từng query_term: log(N / (1 + df)) — từ xuất hiện trong càng
    # nhiều candidate thì trọng số càng thấp, từ hiếm/đặc trưng trọng số cao.
    idf = {}
    for term in query_terms:
        df = sum(1 for doc_terms in doc_terms_list if term in doc_terms)
        idf[term] = math.log((n_docs + 1) / (df + 1)) + 1  # +1 để tránh idf=0 khi từ xuất hiện ở mọi doc

    max_possible = sum(idf.values())  # điểm tối đa nếu 1 doc chứa TẤT CẢ query_terms

    # Raw BM25 (tie-break phụ) — giữ nguyên như cũ
    bm25_scores = None
    try:
        from rank_bm25 import BM25Okapi
        corpus_tokens = [_tokenize(_build_doc_text(row, MAX_CHARS_PER_UNIT)) for row in seeds]
        bm25 = BM25Okapi(corpus_tokens)
        bm25_scores = bm25.get_scores(list(query_terms))
    except Exception as e:
        logger.warning("rerank: rank_bm25 lỗi (%s), bỏ qua tie-break phụ.", e)

    scored = []
    for idx, row in enumerate(seeds):
        doc_terms = doc_terms_list[idx]
        matched_terms = query_terms & doc_terms
        weighted_coverage = sum(idf[t] for t in matched_terms) / max_possible if max_possible > 0 else 0.0

        row = dict(row)
        row["rerank_score"] = weighted_coverage * 10.0
        row["_bm25_tiebreak"] = float(bm25_scores[idx]) if bm25_scores is not None else 0.0
        scored.append(row)

    scored = [r for r in scored if r["rerank_score"] >= min_score]
    scored.sort(key=lambda r: (r["rerank_score"], r["_bm25_tiebreak"]), reverse=True)

    for r in scored:
        r.pop("_bm25_tiebreak", None)

    return scored[:top_n]