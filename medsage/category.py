import re, yaml

def normalize_for_match(text): return re.sub(r'\s+','',str(text)).lower()
def load_lexicon(path):
    with open(path,'r',encoding='utf-8') as f: return yaml.safe_load(f) or {}
def assign_categories(question, lexicon):
    q=normalize_for_match(question); cats=[]
    for cat, terms in lexicon.items():
        if any(normalize_for_match(t) in q for t in terms): cats.append(cat)
    return cats
def extract_terms(text, lexicon):
    t=normalize_for_match(text); found=[]
    for terms in lexicon.values():
        for term in terms:
            if normalize_for_match(term) and normalize_for_match(term) in t: found.append(term)
    return sorted(set(found))
