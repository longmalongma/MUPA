import re


def extract_ans(response: str):
    ans_pat = re.compile(r"2\)\s*Best choice:\s*(.+)", re.S)
    m_ans = ans_pat.search(response)
    answer = m_ans.group(1).strip() if m_ans else None
    if answer == None:
        answer = "(A) None"
    return answer

def resolve_ans(answer: str):
    m_letter = re.match(r"\(([A-Z])\)", answer)
    ans_letter = m_letter.group(1) if m_letter else None
    if m_letter:
        ans_text = answer[m_letter.end():].strip()
    else:
        ans_text = answer
    return ans_letter, ans_text


AUX_VERBS = {
    'am', 'is', 'are', 'was', 'were',
    'do', 'does', 'did',
    'have', 'has', 'had',
    'will', 'would', 'shall', 'should', 'can', 'could', 'may', 'might', 'must'
}


def normalize_answer(ans: str) -> str:
    """
    去掉开头的 Because/Since/As 并去掉末尾标点，小写首字母。
    """
    ans = re.sub(r'^(Because|because|Since|since|As|as)\s+', '', ans)
    ans = ans.rstrip('.!?，。')
    return ans[0].lower() + ans[1:] if ans else ans


def strip_wh_and_aux(question: str) -> str:
    """
    通用地去掉第一个单词（疑问词 what/when/...）和后面可能跟着的助动词/系动词，
    返回剩余的主谓结构字符串（保留后续所有词序）。
    """
    # 清理末尾标点，转小写
    q = question.strip().rstrip('?.！!?').lower()
    tokens = q.split()
    if not tokens:
        return ''
    # 丢弃第一个词（疑问词）
    tokens = tokens[1:]
    # 如果第一个剩余词是助动词/系动词，也去掉
    if tokens and tokens[0] in AUX_VERBS:
        tokens = tokens[1:]
    # 返回剩余部分
    return ' '.join(tokens)


def generate_query(question: str,
                   answer: str,
                   use_question: bool = True,
                   use_answer: bool = True) -> str:
    """
    - 至少 use_question 或 use_answer 为 True，否则抛错
    - 去掉 question 的疑问词和首个助动词/系动词后，得到 core_clause
    - answer 归一化后为 ans_main
    - 如果同时用 Q&A，则:
         “The moment when {core_clause} {ans_main}”
      仅用 Q 则:
         “The moment when {core_clause}”
      仅用 A 则:
         “The moment when {ans_main}”
    """
    if not (use_question or use_answer):
        raise ValueError("generate_query: must use at least question or answer")

    ans_main = normalize_answer(answer) if use_answer else ''
    if use_question:
        core = strip_wh_and_aux(question)
        if use_answer:
            return f"The moment when {core} {ans_main}"
        else:
            return f"The moment when {core}"
    else:
        # 只用 answer
        return f"The moment when {ans_main}"


if __name__ == "__main__":
    qas = [
        {"question": "What fruit did she take out?", "answer": "Kiwi."},
        {"question": "What is done before slicing the leek?", "answer": "Cutting off the other end."},
        {"question": "When does the person slice the leek?", "answer": "After cutting off the other end."},
        {"question": "Who is slicing the leek?", "answer": "The person."},
        {"question": "Which fruit is being peeled?", "answer": "Kiwi."},
        {"question": "How does he chop the beans?", "answer": "Evenly."},
        {"question": "Why does the person go to the pantry?", "answer": "To take the cooking oil."},
        {"question": "Whose hands is the man washing?", "answer": "Her hands."},
        {"question": "Where does the man place the bread?", "answer": "Into the bag."},
    ]

    for qa in qas:
        print(generate_query(qa["question"], qa["answer"],use_question=True,use_answer=True))
