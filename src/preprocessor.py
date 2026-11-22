import re
import unicodedata

def normalize_unicode(text: str) -> str:
    """NFC Unicode normalization for consistent diacritic & character representation."""
    if not isinstance(text, str):
        return ""
    return unicodedata.normalize('NFC', text)

def clean_whitespace(text: str) -> str:
    """Remove special invisible unicode markers, carriage returns, multiple spaces."""
    if not isinstance(text, str):
        return ""
    text = text.replace('\xa0', ' ').replace('\u200b', '').replace('\r', ' ')
    text = re.sub(r'[\t\f\v]+', ' ', text)
    return re.sub(r'\s+', ' ', text).strip()

def preprocess_text(text: str, lowercase: bool = False, remove_punctuation: bool = False) -> str:
    """
    Standard low-resource text preprocessing pipeline:
    1. Unicode NFC normalization
    2. Zero-width & non-breaking space cleansing
    3. Optional lowercasing
    4. Space padding around punctuation for cleaner subword boundaries
    """
    if not isinstance(text, str):
        return ""
    
    text = normalize_unicode(text)
    text = clean_whitespace(text)

    if lowercase:
        text = text.lower()

    if remove_punctuation:
        text = re.sub(r'[^\w\s]', '', text)
    else:
        # Separate punctuation with whitespace for better tokenization
        text = re.sub(r'([.,!?;:"()\[\]{}])', r' \1 ', text)
        text = clean_whitespace(text)

    return text

def extract_features(text: str):
    """
    Feature extraction for low-resource linguistic analysis:
    - Character length
    - Word count
    - Average token length
    - Type-Token Ratio (TTR) - Lexical diversity
    - Non-ASCII / script character ratio
    """
    words = text.split()
    total_tokens = len(words)
    unique_tokens = len(set(words))
    ttr = (unique_tokens / total_tokens) if total_tokens > 0 else 0.0
    
    char_len = len(text)
    avg_tok_len = (sum(len(w) for w in words) / total_tokens) if total_tokens > 0 else 0.0
    
    non_ascii_count = sum(1 for c in text if ord(c) > 127)
    non_ascii_ratio = (non_ascii_count / max(char_len, 1))

    return {
        "char_length": char_len,
        "token_count": total_tokens,
        "unique_tokens": unique_tokens,
        "lexical_diversity_ttr": round(ttr, 3),
        "avg_token_length": round(avg_tok_len, 2),
        "non_ascii_ratio": round(non_ascii_ratio, 3)
    }
