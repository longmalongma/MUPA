# Copyright (c) Huilin Song. Licensed under the BSD 3-Clause License.

import nncore

from MUPA.dataset.hybrid import DATASETS
from MUPA.dataset.wrappers import GroundingDataset
from MUPA.utils.parser import parse_query

import random


@DATASETS.register(name='tacos')
class TACoSDataset(GroundingDataset):
    # use qa with blindqa
    ANNO_PATH_TRAIN_QA_BLINDQA = 'data/tacos_qa/train.jsonl'
    ANNO_PATH_VALID_QA_BLINDQA = 'data/tacos_qa/val.jsonl'
    ANNO_PATH_TEST_QA_BLINDQA = 'data/tacos_qa/test.jsonl'
    # original annos
    ANNO_PATH_TRAIN = 'data/tacos/train.jsonl'
    ANNO_PATH_VALID = 'data/tacos/val.jsonl'
    ANNO_PATH_TEST = 'data/tacos/test.jsonl'

    VIDEO_ROOT = 'data/tacos/videos_3fps_480_noaudio'

    UNIT = 0.001

    @classmethod
    def load_annos(self, split='train', use_qa=False):
        if split == 'train':
            if use_qa:
                raw_annos = nncore.load(self.ANNO_PATH_TRAIN_QA_BLINDQA)
            else:
                raw_annos = nncore.load(self.ANNO_PATH_TRAIN)
        elif split == 'val':
            if use_qa:
                raw_annos = nncore.load(self.ANNO_PATH_VALID_QA_BLINDQA)
            else:
                raw_annos = nncore.load(self.ANNO_PATH_VALID)
        else:
            if use_qa:
                raw_annos = nncore.load(self.ANNO_PATH_TEST_QA_BLINDQA)
            else:
                raw_annos = nncore.load(self.ANNO_PATH_TEST)

        annos = []
        for raw_anno in raw_annos:
            assert len(raw_anno['relevant_windows']) == 1
            vid = raw_anno['vid']
            if use_qa:
                qa_pairs = raw_anno["qa_base"]
                if qa_pairs == 'NA' or not isinstance(qa_pairs, list):
                    continue
                for qa in qa_pairs:
                    question = qa["question"]
                    answer = qa["answer"]
                    distractors = qa["distractor"]
                    ans_options = distractors.copy()
                    ans_options.append(answer)
                    random.shuffle(ans_options)

                    anno = dict(
                        source='tacos',
                        data_type='grounding',
                        video_path=nncore.join(self.VIDEO_ROOT, vid + '-cam-002.mp4'),
                        duration=raw_anno['duration'],
                        query=parse_query(raw_anno['query']),
                        span=raw_anno['relevant_windows'],
                        question=question,
                        options=ans_options,
                        ans=answer
                    )
                    annos.append(anno)
            else:
                anno = dict(
                    source='tacos',
                    data_type='grounding',
                    video_path=nncore.join(self.VIDEO_ROOT, vid + '-cam-002.mp4'),
                    duration=raw_anno['duration'],
                    query=parse_query(raw_anno['query']),
                    span=raw_anno['relevant_windows'],
                )
                annos.append(anno)

        return annos
