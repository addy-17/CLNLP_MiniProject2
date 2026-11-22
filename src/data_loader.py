import os
import re
import openpyxl
import pandas as pd
from src.preprocessor import preprocess_text, extract_features

LANG_MAP = {
    'karbi': {'name': 'Karbi', 'file': 'Category II/English - Karbi Training Data 2026.xlsx', 'category': 'II'},
    'kokborok': {'name': 'Kokborok', 'file': 'Category II/English-Kokborok Training Data 2026.xlsx', 'category': 'II'},
    'tagin': {'name': 'Tagin', 'file': 'Category II/English-Tagin Training Data 2026.xlsx', 'category': 'II'},
    'nagamese': {'name': 'Nagamese', 'file': 'Category II/English-Nagamese  Training Data 2026.xlsx', 'category': 'II'},
    'bodo': {'name': 'Bodo', 'file': 'Category II/English-Bodo Training Data 2026.xlsx', 'category': 'II'},
    'mizo': {'name': 'Mizo', 'file': 'Category I/English-Mizo Traning Data 2026.xlsx', 'category': 'I'},
    'khasi': {'name': 'Khasi', 'file': 'Category I/English-Khasi Training Data 2026.xlsx', 'category': 'I'},
    'manipuri': {'name': 'Manipuri', 'file': 'Category I/English-Manipuri Training Data 2026.xlsx', 'category': 'I'},
    'meitei_mayek': {'name': 'Meitei-Mayek', 'file': 'Category I/English-Meitei-Mayek Transining Data 2026.xlsx', 'category': 'I'},
    'nyishi': {'name': 'Nyishi', 'file': 'Category I/English-Nyishi Training Data 2026.xlsx', 'category': 'I'},
    'assamese': {'name': 'Assamese', 'file': 'Category I/English-Assamese Training Data 2026.csv', 'category': 'I'}
}

def load_language_pair(lang_key, max_samples=1500, lowercase=False):
    if lang_key not in LANG_MAP:
        raise ValueError(f"Language key '{lang_key}' not found in registry.")
    
    info = LANG_MAP[lang_key]
    filepath = info['file']
    if not os.path.exists(filepath):
        filepath = filepath.replace('/', '\\')

    raw_pairs = []
    if filepath.endswith('.csv'):
        df = pd.read_csv(filepath, nrows=max_samples * 2 if max_samples else None)
        for _, row in df.iterrows():
            raw_pairs.append((row['en'], row['as']))
    else:
        wb = openpyxl.load_workbook(filepath, read_only=True)
        sheet = wb.active
        limit = (max_samples * 2) if max_samples else None
        count = 0
        for row in sheet.iter_rows(values_only=True):
            if not row or len(row) < 2:
                continue
            raw_pairs.append((row[0], row[1]))
            count += 1
            if limit and count >= limit:
                break
        wb.close()

    if not raw_pairs:
        return []

    # Detect header presence
    row0_col0 = str(raw_pairs[0][0]).strip().lower()
    row0_col1 = str(raw_pairs[0][1]).strip().lower()
    
    is_header = False
    col_order_inverted = False
    
    if 'eng' in row0_col0 or row0_col0 == 'en':
        is_header = True
    elif 'eng' in row0_col1 or row0_col1 == 'en':
        is_header = True
        col_order_inverted = True
    elif any(lang in row0_col0 for lang in ['tagin', 'karbi', 'bodo', 'khasi', 'manipuri']):
        is_header = True
        col_order_inverted = True

    start_idx = 1 if is_header else 0

    valid_pairs = []
    for r in raw_pairs[start_idx:]:
        raw_s, raw_t = (r[1], r[0]) if col_order_inverted else (r[0], r[1])
        
        # Apply preprocessing pipeline
        src = preprocess_text(raw_s, lowercase=lowercase)
        tgt = preprocess_text(raw_t, lowercase=lowercase)

        if len(src) >= 2 and len(tgt) >= 2:
            src_words = src.split()
            tgt_words = tgt.split()
            if 1 <= len(src_words) <= 50 and 1 <= len(tgt_words) <= 50:
                valid_pairs.append((src, tgt))

    if max_samples and len(valid_pairs) > max_samples:
        valid_pairs = valid_pairs[:max_samples]

    return valid_pairs

def get_corpus_statistics(lang_key, sample_size=1000):
    pairs = load_language_pair(lang_key, max_samples=sample_size)
    if not pairs:
        return {}

    src_texts = [p[0] for p in pairs]
    tgt_texts = [p[1] for p in pairs]

    src_stats = [extract_features(t) for t in src_texts]
    tgt_stats = [extract_features(t) for t in tgt_texts]

    def avg_metric(lst, key):
        return round(sum(item[key] for item in lst) / max(len(lst), 1), 2)

    return {
        "lang_key": lang_key,
        "sample_count": len(pairs),
        "src_avg_tokens": avg_metric(src_stats, "token_count"),
        "tgt_avg_tokens": avg_metric(tgt_stats, "token_count"),
        "src_lexical_diversity": avg_metric(src_stats, "lexical_diversity_ttr"),
        "tgt_lexical_diversity": avg_metric(tgt_stats, "lexical_diversity_ttr"),
        "src_avg_word_length": avg_metric(src_stats, "avg_token_length"),
        "tgt_avg_word_length": avg_metric(tgt_stats, "avg_token_length"),
        "tgt_script_density": avg_metric(tgt_stats, "non_ascii_ratio")
    }

def get_available_languages():
    return [{
        'key': k,
        'name': v['name'],
        'category': v['category'],
        'file': v['file']
    } for k, v in LANG_MAP.items()]
