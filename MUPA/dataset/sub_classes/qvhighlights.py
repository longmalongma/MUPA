import nncore

from MUPA.dataset.hybrid import DATASETS
from MUPA.dataset.wrappers import GroundingDataset
from MUPA.utils.parser import parse_query

import random


@DATASETS.register(name='qvhighlights')
class QVHighlightsDataset(GroundingDataset):

    ANNO_PATH_TRAIN_QA_BLINDQA = 'data/qvhighlights_qa/highlight_train_release.jsonl'
    ANNO_PATH_VALID_QA_BLINDQA = 'data/qvhighlights_qa/highlight_val_release.jsonl'
    ANNO_PATH_TEST_QA_BLINDQA = 'data/qvhighlights_qa/highlight_test_release.jsonl'

    ANNO_PATH_TRAIN = 'data/qvhighlights/highlight_train_release.jsonl'
    ANNO_PATH_VALID = 'data/qvhighlights/highlight_val_release.jsonl'
    ANNO_PATH_TEST = 'data/qvhighlights/highlight_test_release.jsonl'

    VIDEO_ROOT = 'data/qvhighlight/videos_3fps_480_noaudio'

    UNIT = 2.0

    @classmethod
    def load_annos(self, split='train',use_qa=False):
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
            vid = raw_anno['vid']
            qid = raw_anno['qid']

            anno = dict(
                source='qvhighlights',
                data_type='grounding',
                video_path=nncore.join(self.VIDEO_ROOT, vid + '.mp4'),
                duration=raw_anno['duration'],
                query=parse_query(raw_anno['query']),
                span=raw_anno.get('relevant_windows'),
                vid=vid,
                qid=qid
            )

            annos.append(anno)

        return annos


@DATASETS.register(name='qvhighlights_single')
class QVHighlightsSingleDataset(QVHighlightsDataset):

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
            # skip samples with multiple moments
            if len(raw_anno['relevant_windows']) > 1:
                continue
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
                        source='qvhighlights_single',
                        data_type='grounding',
                        video_path=nncore.join(self.VIDEO_ROOT, vid + '.mp4'),
                        duration=raw_anno['duration'],
                        query=parse_query(raw_anno['query']),
                        span=raw_anno.get('relevant_windows'),
                        question=question,
                        options=ans_options,
                        ans=answer
                    )
                    annos.append(anno)
            else:
                anno = dict(
                    source='qvhighlights_single',
                    data_type='grounding',
                    video_path=nncore.join(self.VIDEO_ROOT, vid + '.mp4'),
                    duration=raw_anno['duration'],
                    query=parse_query(raw_anno['query']),
                    span=raw_anno.get('relevant_windows')
                )
                annos.append(anno)

        return annos
