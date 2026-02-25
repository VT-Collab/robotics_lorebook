from termcolor import cprint
from .main_utils import generate_rag_key

def retrieve_feedback_context(env, gen, lorebook, messages, subtask, verbose) -> str:
    rag_query_key = generate_rag_key(env, subtask)
    tmp_retrieved_lore = lorebook.query(rag_query_key, top_k=20)
    retrieved_lore = []
    for item in tmp_retrieved_lore:
        if "GENERAL" in item["key"] or "TEMPLATE" in item["key"]:
            continue
        else: 
            retrieved_lore.append(item)

    cprint(f"n lore: {len(retrieved_lore)}")

    if not retrieved_lore:
        return ""

    cprint("Integrating past feedback...", "cyan")
    feedback_items = "\n".join([f"- {item['value']}" for item in retrieved_lore])
    raw_context = f"Use the following past experience as feedback:\n{feedback_items}"

    return f" You should refer to human's feedback to accomplish the task:\n{raw_context}"