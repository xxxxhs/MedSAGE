from .io import read_csv_auto

def load_cmedqa2(question_file, answer_file, question_id_col='question_id', question_text_col='content', answer_question_id_col='question_id', answer_text_col='content'):
    qdf = read_csv_auto(question_file); adf = read_csv_auto(answer_file)
    if not {question_id_col, question_text_col}.issubset(qdf.columns):
        raise ValueError(f'question.csv must contain {question_id_col}, {question_text_col}')
    if not {answer_question_id_col, answer_text_col}.issubset(adf.columns):
        raise ValueError(f'answer.csv must contain {answer_question_id_col}, {answer_text_col}')
    qdf = qdf[[question_id_col, question_text_col]].dropna().copy()
    adf = adf[[answer_question_id_col, answer_text_col]].dropna().copy()
    qdf[question_id_col] = qdf[question_id_col].astype(str)
    adf[answer_question_id_col] = adf[answer_question_id_col].astype(str)
    qdf = qdf.drop_duplicates(subset=[question_id_col])
    return qdf, adf

def build_qid_to_answers(answer_df, qid_col='question_id', answer_col='content'):
    qid_to_answers = {}
    for _, row in answer_df.iterrows():
        qid = str(row[qid_col]); ans = str(row[answer_col]).strip()
        if ans: qid_to_answers.setdefault(qid, []).append(ans)
    return qid_to_answers

def make_multi_answer_rows(question_df, qid_to_answers, qid_col='question_id', question_col='content', min_answers=2):
    rows=[]
    for _, row in question_df.iterrows():
        qid=str(row[qid_col]); answers=qid_to_answers.get(qid, [])
        if len(answers) >= min_answers:
            rows.append({'question_id': qid, 'question': str(row[question_col]).strip(), 'answers': answers, 'num_answers': len(answers)})
    return rows
