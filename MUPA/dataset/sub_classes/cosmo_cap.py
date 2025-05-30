# Copyright (c) Huilin Song. Licensed under the BSD 3-Clause License.

import nncore

from MUPA.dataset.hybrid import DATASETS
from MUPA.dataset.wrappers import GroundingDataset
from MUPA.utils.parser import parse_query

import random

@DATASETS.register(name='cosmo_cap')
class CosMoCapDataset(GroundingDataset):

    ANNO_PATH_QA_BLINDQA = 'data/cosmo_cap_qa/anno_cosmo_cap.jsonl'

    ANNO_PATH = 'data/cosmo_cap/anno_cosmo_cap.jsonl'

    VIDEO_ROOT = 'data/cosmo_cap/videos_3fps_480_noaudio'

    UNIT = 1.0

    @classmethod
    def load_annos(self, split='train', use_qa=False):
    
        assert split == 'train'
        if use_qa:
            raw_annos = nncore.load(self.ANNO_PATH_QA_BLINDQA)
        else:
            raw_annos = nncore.load(self.ANNO_PATH)

        annos = []
        for raw_anno in raw_annos:
            if use_qa:
                qa_pairs = raw_anno["qa_base"]
                if qa_pairs == 'NA' or not isinstance(qa_pairs, list):
                    continue
                for qa in qa_pairs:  # 对每个QApair单独做一个样本
                    question = qa["question"]
                    answer = qa["answer"]
                    distractors = qa["distractor"]
                    ans_options = distractors.copy()
                    ans_options.append(answer)
                    random.shuffle(ans_options)
                    anno = dict(
                        source='cosmo_cap',
                        data_type='grounding',
                        video_path=nncore.join(self.VIDEO_ROOT, raw_anno['vid'] + '.mp4'),
                        duration=raw_anno['duration'],
                        query=parse_query(raw_anno['query']),
                        span=[raw_anno['span']],
                        question=question,
                        options=ans_options,
                        ans=answer
                    )
                    annos.append(anno)
            else:
                anno = dict(
                    source='cosmo_cap',
                    data_type='grounding',
                    video_path=nncore.join(self.VIDEO_ROOT, raw_anno['vid'] + '.mp4'),
                    duration=raw_anno['duration'],
                    query=parse_query(raw_anno['query']),
                    span=[raw_anno['span']],
                )
                annos.append(anno)


        return annos
