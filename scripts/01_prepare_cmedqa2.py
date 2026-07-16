from __future__ import annotations
import argparse
from pathlib import Path
import sys
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from medsage.config import load_config, ensure_dir
from medsage.data import load_cmedqa2, build_qid_to_answers, make_multi_answer_rows
from medsage.io import write_json, write_jsonl

def main():
    p=argparse.ArgumentParser(); p.add_argument('--config',default='configs/default.yaml'); p.add_argument('--paths',default='configs/paths.yaml'); args=p.parse_args()
    cfg=load_config(args.config,args.paths)
    qdf,adf=load_cmedqa2(cfg['data']['question_file'],cfg['data']['answer_file'],cfg['data'].get('question_id_col','question_id'),cfg['data'].get('question_text_col','content'),cfg['data'].get('answer_question_id_col','question_id'),cfg['data'].get('answer_text_col','content'))
    qid_to_answers=build_qid_to_answers(adf,cfg['data'].get('answer_question_id_col','question_id'),cfg['data'].get('answer_text_col','content'))
    rows=make_multi_answer_rows(qdf,qid_to_answers,cfg['data'].get('question_id_col','question_id'),cfg['data'].get('question_text_col','content'),int(cfg['data'].get('min_answers_per_question',2)))
    ensure_dir(Path(cfg['outputs']['prepared_questions']).parent); write_jsonl(cfg['outputs']['prepared_questions'], rows); write_json(cfg['outputs']['qid_to_answers'], qid_to_answers)
    print(f'Loaded questions: {len(qdf)}'); print(f'Loaded answers: {len(adf)}'); print(f'Multi-answer questions saved: {len(rows)}')
if __name__=='__main__': main()
