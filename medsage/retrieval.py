from pathlib import Path

def build_faiss_index(embeddings, normalize=True):
    import faiss
    embeddings = embeddings.astype('float32')
    if normalize: faiss.normalize_L2(embeddings)
    index = faiss.IndexFlatIP(embeddings.shape[1]); index.add(embeddings); return index

def save_faiss_index(index, path):
    import faiss
    path=Path(path); path.parent.mkdir(parents=True, exist_ok=True); faiss.write_index(index, str(path))

def load_faiss_index(path):
    import faiss
    return faiss.read_index(str(path))

def search_index(index, query_embeddings, top_k=50, normalize=True):
    import faiss
    query_embeddings=query_embeddings.astype('float32')
    if normalize: faiss.normalize_L2(query_embeddings)
    return index.search(query_embeddings, top_k)

def collect_retrieved_answers(query_id, retrieved_indices, retrieved_scores, indexed_qids, qid_to_answers, remove_same_question_id=True, max_records=50):
    records=[]; seen=set()
    for idx, score in zip(retrieved_indices, retrieved_scores):
        if idx < 0 or idx >= len(indexed_qids): continue
        qid=str(indexed_qids[idx])
        if remove_same_question_id and qid == str(query_id): continue
        if qid in seen: continue
        seen.add(qid); answers=qid_to_answers.get(qid, [])
        if not answers: continue
        records.append({'question_id': qid, 'score': float(score), 'answers': answers})
        if len(records) >= max_records: break
    return {'query_id': str(query_id), 'retrieved': records}
