def add_bertscore_best_match(rows, model_path=None, lang='zh'):
    from bert_score import score as bert_score
    out=[]
    for row in rows:
        cand=row.get('generated_answer',''); refs=row.get('references',[]) or []
        if refs:
            kwargs={'lang':lang,'verbose':False,'rescale_with_baseline':False}
            if model_path: kwargs['model_type']=model_path
            _,_,f1=bert_score([cand]*len(refs), refs, **kwargs); val=float(f1.max().item())
        else: val=0.0
        r=dict(row); r['bertscore_f1']=val; out.append(r)
    return out
