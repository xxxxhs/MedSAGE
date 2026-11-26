# Prompt templates documented in Multimedia Appendix 1, Table S6.
BASE_SYSTEM_PROMPT = '你是一名专业医生，负责回答中文医学健康咨询问题。请给出科学、准确、完整、通俗的回答。结合已提供的医生证据，但不应无条件照搬错误建议。未能确认的事实说明不确定性；必要时给出就医提示。目标约450汉字，允许400—500汉字。直接输出正文，不要重复题目。'
EVIDENCE_SYSTEM_PROMPT = BASE_SYSTEM_PROMPT

def build_base_prompt(question):
    return [{'role': 'system', 'content': BASE_SYSTEM_PROMPT},
            {'role': 'user', 'content': f'患者问题：{question}\n\n请直接回答该问题。'}]

def build_evidence_prompt(question, evidence_sentences):
    evidence = '\n'.join(f'[证据 {i+1}] {s}' for i, s in enumerate(evidence_sentences))
    prompt = f'患者问题：{question}\n\n医学专家知识库检索证据：\n\n{evidence}\n\n请综合以上证据作答。'
    return [{'role': 'system', 'content': EVIDENCE_SYSTEM_PROMPT},
            {'role': 'user', 'content': prompt}]

def messages_to_plain_text(messages):
    return '\n\n'.join(f"{m['role'].upper()}:\n{m['content']}" for m in messages)
