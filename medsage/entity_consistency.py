from .category import extract_terms

def entity_consistency_for_row(row, lexicon):
    pred=set(extract_terms(row.get('generated_answer',''), lexicon))
    ref=set(extract_terms(row.get('question','')+' '+' '.join(row.get('references',[]) or []), lexicon))
    supported=pred & ref; unsupported=pred-ref
    precision=len(supported)/len(pred) if pred else 0.0; recall=len(supported)/len(ref) if ref else 0.0
    f1=2*precision*recall/(precision+recall) if precision+recall>0 else 0.0
    out=dict(row); out.update({'entity_precision':precision,'entity_recall':recall,'entity_f1':f1,'unsupported_entity_rate':len(unsupported)/len(pred) if pred else 0.0,'pred_entities':sorted(pred),'ref_entities':sorted(ref),'unsupported_entities':sorted(unsupported),'mean_pred_entities':len(pred),'mean_unsupported_entities':len(unsupported)})
    return out

def summarize_entity_consistency(rows):
    keys=['entity_precision','entity_recall','entity_f1','unsupported_entity_rate','mean_pred_entities','mean_unsupported_entities']
    return {k: sum(float(r.get(k,0.0)) for r in rows)/len(rows) if rows else 0.0 for k in keys}
