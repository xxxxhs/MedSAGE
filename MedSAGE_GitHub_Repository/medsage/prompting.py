BASE_SYSTEM_PROMPT = '你是一名专业医生，你需要根据用户的问题给出适当的回答。'
EVIDENCE_SYSTEM_PROMPT = '你是一名专业医生，你需要根据用户的问题与多名专家的已有回答给出适当的回答。'

def build_base_prompt(question):
    return [{'role':'system','content':BASE_SYSTEM_PROMPT},{'role':'user','content':f'用户问题为：{question}\n\n请根据用户问题，给出适当、清楚、谨慎的医学回答。'}]

def build_evidence_prompt(question, evidence_sentences):
    evidence='\n'.join([f'{i+1}. {s}' for i,s in enumerate(evidence_sentences)])
    prompt=f'用户问题为：{question}\n\n专家已有回答为：\n{evidence}\n\n请根据用户问题与上述专家已有回答，给出适当、清楚、谨慎的医学回答。'
    return [{'role':'system','content':EVIDENCE_SYSTEM_PROMPT},{'role':'user','content':prompt}]

def messages_to_plain_text(messages): return '\n\n'.join([f"{m['role'].upper()}:\n{m['content']}" for m in messages])
