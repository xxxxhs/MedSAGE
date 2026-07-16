from collections import Counter
import math, numpy as np

def tokenize_zh(text): return [c for c in str(text).strip() if not c.isspace()]
def ngrams(tokens,n): return [tuple(tokens[i:i+n]) for i in range(len(tokens)-n+1)]

def bleu2(candidate, reference):
    cand=tokenize_zh(candidate); ref=tokenize_zh(reference)
    if not cand or not ref: return 0.0
    ps=[]
    for n in [1,2]:
        c=Counter(ngrams(cand,n)); r=Counter(ngrams(ref,n))
        overlap=sum(min(v,r[k]) for k,v in c.items()); ps.append((overlap+1e-9)/(sum(c.values())+1e-9))
    bp=1.0 if len(cand)>len(ref) else math.exp(1-len(ref)/max(len(cand),1))
    return bp*math.exp(sum(math.log(p+1e-9) for p in ps)/2)

def lcs_len(a,b):
    dp=[0]*(len(b)+1)
    for x in a:
        prev=0
        for j,y in enumerate(b,1):
            tmp=dp[j]; dp[j]=prev+1 if x==y else max(dp[j], dp[j-1]); prev=tmp
    return dp[-1]

def rouge_l(candidate, reference):
    cand=tokenize_zh(candidate); ref=tokenize_zh(reference)
    if not cand or not ref: return 0.0
    l=lcs_len(cand,ref); p=l/len(cand); r=l/len(ref)
    return 0.0 if p+r==0 else 2*p*r/(p+r)

def meteor_lite(candidate, reference):
    cand=tokenize_zh(candidate); ref=tokenize_zh(reference)
    if not cand or not ref: return 0.0
    c=Counter(cand); r=Counter(ref); overlap=sum(min(c[t],r[t]) for t in c)
    p=overlap/max(len(cand),1); rec=overlap/max(len(ref),1)
    return 0.0 if p+rec==0 else (10*p*rec)/(rec+9*p+1e-9)

def cider_lite(candidate, references, n_max=4):
    cand=tokenize_zh(candidate); refs=[tokenize_zh(r) for r in references if str(r).strip()]
    if not cand or not refs: return 0.0
    vals=[]
    for n in range(1,n_max+1):
        cc=Counter(ngrams(cand,n)); rcs=[Counter(ngrams(r,n)) for r in refs]
        vocab=set(cc)
        for rc in rcs: vocab.update(rc)
        if not vocab: continue
        vocab=list(vocab); docs=[cc]+rcs; idf={t: math.log((len(docs)+1)/(sum(1 for d in docs if t in d)+1))+1 for t in vocab}
        def vec(counts): return np.array([counts.get(t,0)*idf[t] for t in vocab], dtype='float32')
        cv=vec(cc); rv=np.mean([vec(rc) for rc in rcs], axis=0); vals.append(float(np.dot(cv,rv)/(np.linalg.norm(cv)*np.linalg.norm(rv)+1e-9)))
    return float(np.mean(vals)) if vals else 0.0

def distinct_n(texts,n=1):
    allng=[]
    for t in texts: allng.extend(ngrams(tokenize_zh(t),n))
    return len(set(allng))/len(allng) if allng else 0.0

def best_match_score(candidate, refs, fn): return max([fn(candidate,r) for r in refs], default=0.0)

def evaluate_generation_rows(rows):
    out=[]
    for row in rows:
        cand=row.get('generated_answer',''); refs=row.get('references',[]) or []; r=dict(row)
        r['bleu2']=best_match_score(cand,refs,bleu2); r['rouge_l']=best_match_score(cand,refs,rouge_l); r['meteor']=best_match_score(cand,refs,meteor_lite); r['cider']=cider_lite(cand,refs); out.append(r)
    d1=distinct_n([r.get('generated_answer','') for r in rows],1); d2=distinct_n([r.get('generated_answer','') for r in rows],2)
    for r in out: r['distinct1_corpus']=d1; r['distinct2_corpus']=d2
    return out

def summarize_metrics(rows, keys): return {k: float(np.mean([float(r[k]) for r in rows if k in r])) if any(k in r for r in rows) else 0.0 for k in keys}
