import numpy as np

def holm_correction(p_values):
    m=len(p_values); order=np.argsort(p_values); adjusted=np.empty(m); prev=0.0
    for rank,idx in enumerate(order):
        adj=max((m-rank)*p_values[idx], prev); adjusted[idx]=min(adj,1.0); prev=adjusted[idx]
    return adjusted.tolist()

def paired_wilcoxon(x,y):
    from scipy.stats import wilcoxon
    if len(x)==0: return 1.0
    if np.allclose(np.asarray(y)-np.asarray(x),0): return 1.0
    return float(wilcoxon(x,y,zero_method='wilcox',alternative='two-sided').pvalue)

def compare_methods(method_rows, target_method, baseline_methods, metrics):
    records=[]; pvals=[]
    for baseline in baseline_methods:
        common=sorted(set(method_rows[target_method]) & set(method_rows[baseline]))
        for metric in metrics:
            base=[float(method_rows[baseline][qid][metric]) for qid in common if metric in method_rows[baseline][qid] and metric in method_rows[target_method][qid]]
            targ=[float(method_rows[target_method][qid][metric]) for qid in common if metric in method_rows[baseline][qid] and metric in method_rows[target_method][qid]]
            p=paired_wilcoxon(base,targ); pvals.append(p)
            records.append({'comparison':f'{target_method} vs {baseline}','metric':metric,'comparator_mean':float(np.mean(base)) if base else 0.0,'target_mean':float(np.mean(targ)) if targ else 0.0,'mean_difference':float(np.mean(targ)-np.mean(base)) if base else 0.0,'wilcoxon_p':p,'n':len(base)})
    for rec,adj in zip(records, holm_correction(pvals)):
        rec['holm_adjusted_p']=adj; rec['significance']='***' if adj<0.001 else ('**' if adj<0.01 else ('*' if adj<0.05 else 'ns'))
    return records
