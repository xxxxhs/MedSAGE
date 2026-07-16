import numpy as np
from .embedding import cosine_matrix, cosine_vector

def minmax_normalize(x):
    x=np.asarray(x, dtype='float32')
    if x.size == 0: return x
    mn=float(x.min()); mx=float(x.max())
    return np.ones_like(x) if abs(mx-mn)<1e-12 else (x-mn)/(mx-mn)

def compute_centrality(sentence_embeddings, tau=0.7):
    sim=cosine_matrix(sentence_embeddings); adj=(sim>=tau).astype('float32'); np.fill_diagonal(adj,0.0); return adj.sum(axis=1)

def compute_relevance(sentence_embeddings, query_embedding): return cosine_vector(sentence_embeddings, query_embedding)

def greedy_diversity_scores(sentence_embeddings, selected_indices):
    n=sentence_embeddings.shape[0]
    if not selected_indices: return np.ones(n, dtype='float32')
    max_overlap=cosine_matrix(sentence_embeddings, sentence_embeddings[selected_indices]).max(axis=1)
    return 1.0 - max_overlap

def combine_scores(c, r, d, weights=(1,1,1), normalize_components=True):
    if normalize_components: c,r,d = minmax_normalize(c), minmax_normalize(r), minmax_normalize(d)
    wc,wr,wd=weights; return wc*c + wr*r + wd*d

def select_medsage_sentences(sentences, sentence_embeddings, query_embedding, top_k=10, tau=0.7, weights=(1,1,1), normalize_components=True):
    n=len(sentences)
    if n==0: return {'selected': [], 'scores': []}
    top_k=min(top_k,n); c=compute_centrality(sentence_embeddings,tau); r=compute_relevance(sentence_embeddings,query_embedding)
    selected=[]
    for _ in range(top_k):
        d=greedy_diversity_scores(sentence_embeddings, selected)
        score=combine_scores(c,r,d,weights,normalize_components)
        for i in selected: score[i] = -1e9
        best=int(np.argmax(score)); selected.append(best)
    d=greedy_diversity_scores(sentence_embeddings, selected[:-1] if len(selected)>1 else [])
    final=combine_scores(c,r,d,weights,normalize_components)
    rows=[{'index': int(i), 'sentence': sentences[int(i)], 'centrality': float(c[int(i)]), 'relevance': float(r[int(i)]), 'diversity': float(d[int(i)]), 'score': float(final[int(i)])} for i in selected]
    rows=sorted(rows, key=lambda x: x['score'], reverse=True)
    return {'selected': [x['sentence'] for x in rows], 'scores': rows}

def select_rag_sentences(sentences, top_k=10):
    selected=sentences[:min(top_k,len(sentences))]
    return {'selected': selected, 'scores': [{'index':i,'sentence':s,'score':float(len(selected)-i)} for i,s in enumerate(selected)]}

def select_avg_sentences(sentences, sentence_embeddings, query_embedding, top_k=10):
    if not sentences: return {'selected': [], 'scores': []}
    centroid=np.mean(np.vstack([sentence_embeddings, query_embedding.reshape(1,-1)]), axis=0)
    scores=cosine_vector(sentence_embeddings, centroid); order=np.argsort(-scores)[:min(top_k,len(sentences))]
    rows=[{'index': int(i), 'sentence': sentences[int(i)], 'score': float(scores[int(i)])} for i in order]
    return {'selected': [r['sentence'] for r in rows], 'scores': rows}

def select_mmr_sentences(sentences, sentence_embeddings, query_embedding, top_k=10, lambda_mult=0.7):
    n=len(sentences)
    if n==0: return {'selected': [], 'scores': []}
    top_k=min(top_k,n); relevance=cosine_vector(sentence_embeddings,query_embedding); selected=[]; candidates=set(range(n)); rows=[]
    for _ in range(top_k):
        best_idx=None; best_score=-1e9
        for idx in candidates:
            redundancy=cosine_matrix(sentence_embeddings[idx:idx+1], sentence_embeddings[selected]).max() if selected else 0.0
            score=lambda_mult*relevance[idx] - (1-lambda_mult)*redundancy
            if score>best_score: best_score=float(score); best_idx=idx
        selected.append(best_idx); candidates.remove(best_idx); rows.append({'index': int(best_idx), 'sentence': sentences[int(best_idx)], 'relevance': float(relevance[int(best_idx)]), 'score': float(best_score)})
    return {'selected': [r['sentence'] for r in rows], 'scores': rows}
