from __future__ import annotations
import argparse
from pathlib import Path
import sys
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

# Runs MedSAGE component ablation variants: R_only, C_only, D_only, RC, RD, CD, Full_CRD.
# Use --dry-run to produce prompts without LLM inference.
from pathlib import Path
from tqdm import tqdm
from medsage.config import load_config, ensure_dir
from medsage.embedding import SentenceEmbedder
from medsage.io import read_jsonl, write_jsonl
from medsage.prompting import build_evidence_prompt, messages_to_plain_text
from medsage.segmentation import split_many
from medsage.aggregation import select_medsage_sentences
from medsage.generation import LocalHFGenerator, GenerationConfig

def main():
    p=argparse.ArgumentParser(); p.add_argument('--config',default='configs/default.yaml'); p.add_argument('--paths',default='configs/paths.yaml'); p.add_argument('--model',default=None); p.add_argument('--dry-run',action='store_true'); args=p.parse_args(); cfg=load_config(args.config,args.paths); model=args.model or cfg['models']['default_generation_model']
    rows=read_jsonl(cfg['outputs']['retrieval_results']); embedder=SentenceEmbedder(cfg['models']['embedding_model_path'],cfg['runtime'].get('device','auto'))
    generator=None
    if not args.dry_run:
        mc=cfg['models'][model]; generator=LocalHFGenerator(mc['model_path'],mc.get('tokenizer_path',mc['model_path'])); gen_cfg=GenerationConfig(float(cfg['generation'].get('temperature',0.7)),float(cfg['generation'].get('top_p',0.9)),int(cfg['generation'].get('max_new_tokens',1024)),bool(cfg['generation'].get('do_sample',True)))
    out_dir=Path(cfg['outputs']['generation_dir'])/model/'ablation'; ensure_dir(out_dir)
    for name,w in cfg['ablation']['variants'].items():
        outs=[]; weights=(float(w['centrality']),float(w['relevance']),float(w['diversity']))
        for item in tqdm(rows, desc=name):
            answers=[]
            for rec in item.get('retrieved',[]): answers.extend(rec.get('answers',[]))
            candidates=split_many(answers,cfg['sentence'].get('regex',r'[。！？!?；;]+'),int(cfg['sentence'].get('min_sentence_length',4)),int(cfg['sentence'].get('max_sentence_length',180)))
            if candidates:
                sent=embedder.encode(candidates,show_progress_bar=False); q=embedder.encode([item['question']],show_progress_bar=False)[0]
                evidence=select_medsage_sentences(candidates,sent,q,int(cfg['medsage']['sentence_top_k']),float(cfg['medsage']['similarity_threshold_tau']),weights)['selected']
            else: evidence=[]
            messages=build_evidence_prompt(item['question'],evidence); answer='' if args.dry_run else generator.generate(messages,gen_cfg)
            outs.append({'question_id':item['query_id'],'question':item['question'],'scheme':name,'model':model,'evidence':evidence,'prompt':messages_to_plain_text(messages),'generated_answer':answer,'references':item.get('references',[])})
        write_jsonl(out_dir/f'{name}.jsonl',outs)
if __name__=='__main__': main()
