# Copyright (c) Huilin Song. Licensed under the BSD 3-Clause License.

import copy

from torch.utils.data import Dataset

from MUPA.constants import GROUNDER_PROMPT, GQA_PROMPT, REG_TOKEN
import random


class GroundingDataset(Dataset):

    def __init__(self, processor, model_args, data_args, training_args):
        super(GroundingDataset, self).__init__()

        raw_annos = self.load_annos(use_qa=data_args.use_qa)

        annos = []
        if data_args.use_qa:
            for raw_anno in raw_annos:
                question = raw_anno['question']
                num_words = len(question.split())  # min_num_words=-1, max_num_words=200
                if data_args.min_video_len >= 0 and raw_anno.get('duration', float('inf')) < data_args.min_video_len:
                    continue
                if data_args.max_video_len >= 0 and raw_anno.get('duration', 0) > data_args.max_video_len:
                    continue
                if data_args.min_num_words >= 0 and num_words < data_args.min_num_words:
                    continue
                if data_args.max_num_words >= 0 and num_words > data_args.max_num_words:
                    continue
                annos.append(raw_anno)
        else:
            for anno in raw_annos:
                num_words = len(anno['query'].split(' '))
                if data_args.min_num_words >= 0 and num_words < data_args.min_num_words:
                    continue
                if data_args.max_num_words >= 0 and num_words > data_args.max_num_words:
                    continue
                if data_args.min_video_len >= 0 and anno.get('duration', float('inf')) < data_args.min_video_len:
                    continue
                if data_args.max_video_len >= 0 and anno.get('duration', 0) > data_args.max_video_len:
                    continue
                annos.append(anno)

        self.annos = annos
        self.raw_length = len(raw_annos)
        self.processor = processor
        self.model_args = model_args
        self.data_args = data_args
        self.training_args = training_args

    def __len__(self):
        return len(self.annos)

    def __getitem__(self, idx):
        anno = copy.deepcopy(self.annos[idx])
        video_path, duration, query, span = anno['video_path'], anno['duration'], anno['query'], anno['span']
        question = anno['question']
        options = anno["options"]
        answer = anno["ans"]
        # GQA train
        if self.data_args.use_qa and question is not None and options is not None and answer is not None:
            options_prompt = "Options:"
            for idx, opt in enumerate(options):
                letter = chr(ord("A") + idx)
                options_prompt += f"\n({letter}) {opt.capitalize()}"
            # 找到正确答案在打乱后列表中的索引和对应字母
            ans_index = options.index(answer)
            ans_option = f"({chr(ord('A') + ans_index)})"
            format_content = (
                f"1) The relevant moment happens in {REG_TOKEN}\n"
                f"2) Best choice: {ans_option} {answer}\n"
            )
            messages = [{
                'role': 'user',
                'content': [{
                    'type': 'video',
                    'video': video_path,
                    'min_pixels': 36 * 28 * 28,
                    'max_pixels': 64 * 28 * 28,
                    'max_frames': 150,  # 原150
                    'fps': 1.0,
                }, {
                    'type': 'text',
                    'text': GQA_PROMPT.format(question, options_prompt)
                }]
            }, {
                'role': 'assistant',
                'content': format_content
            }]
        else: # grounder train
            messages = [{
                'role': 'user',
                'content': [{
                    'type': 'video',
                    'video': video_path,
                    'min_pixels': 36 * 28 * 28,
                    'max_pixels': 64 * 28 * 28,
                    'max_frames': 150,
                    'fps': 1.0,
                }, {
                    'type': 'text',
                    'text': GROUNDER_PROMPT.format(query)
                }]
            }, {
                'role': 'assistant',
                'content': f'The relevant moment happens in {REG_TOKEN}.'
            }]

        meta = dict(messages=messages, span=span, duration=duration)

        return meta



