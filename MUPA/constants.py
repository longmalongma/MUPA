# Copyright (c) Huilin Song. Licensed under the BSD 3-Clause License.

IGNORE_INDEX = -100

REG_TOKEN = '<|reg|>'

SEG_S_TOKEN = '<|seg_start|>'
SEG_E_TOKEN = '<|seg_end|>'

GROUNDER_PROMPT = (
    'You are acting as the grounder now. '
    'Given a video and a text query, your goal is to temporally localize the video moment described by the query. '
    'If the query is directly describing a moment, simply localize it according to its content. '
    "Otherwise, if the moment is described as 'before/after a pivotal event', you need to determine the actual event it refers to. "
    'The localized moment should only cover the target event. '
    "Now I give you the query: '{}'. "
    'Please think carefully and provide your response.')

GQA_PROMPT = (
    "You are acting as the GQA Agent now.\n"
    "Given a video and a multiple‑choice question, you have two tasks:\n"
    f"1) Trigger the video‑moment retrieve pipeline to temporally localize the video moment described by the question by generating exactly {REG_TOKEN}.\n"
    f"2) Choose the best answer from given options.\n"
    "Question: {}\n"
    "Options:\n{}\n"
    "Please reply exactly in this format:\n"
    f"1) The relevant moment happens in {REG_TOKEN}\n"
    f"2) Best choice: <Option>\n"
)

VERIFIER_PROMPT = (
    'You are acting as the verifier now. '
    'You will be presented a text query describing a moment that potentialy happens in the given video. '
    f'Your task is to identify whether the video segment between {SEG_S_TOKEN} and {SEG_E_TOKEN} perfectly covers the moment. '
    f'If the described moment can be seen in the video, please focus on verifying whether the moment starts at {SEG_S_TOKEN} and ends at {SEG_E_TOKEN}. '
    "Respond with 'Yes' if you think the moment boundaries are correct, otherwise 'No'. "
    "If the described moment cannot be seen in the video, respond with 'No' directly. "
    "Now I give you the query: '{}'. "
    "Please think carefully and respond with 'Yes' or 'No' directly.")
