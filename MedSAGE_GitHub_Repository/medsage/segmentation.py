import re
DEFAULT_SENTENCE_REGEX = r'[。！？!?；;]+'

def normalize_text(text):
    text = '' if text is None else str(text)
    return re.sub(r'\s+', ' ', text).strip()

def split_sentences(text, regex=DEFAULT_SENTENCE_REGEX, min_len=4, max_len=180):
    text = normalize_text(text)
    out=[]
    for frag in re.split(regex, text):
        frag=normalize_text(frag)
        if len(frag) < min_len: continue
        if len(frag) > max_len: continue
        out.append(frag)
    return out

def split_many(texts, regex=DEFAULT_SENTENCE_REGEX, min_len=4, max_len=180):
    s=[]
    for text in texts: s.extend(split_sentences(text, regex, min_len, max_len))
    return s
