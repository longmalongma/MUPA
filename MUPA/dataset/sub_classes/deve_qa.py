import nncore
from MUPA.dataset.hybrid import DATASETS
from MUPA.dataset.wrappers import GroundingDataset
import random


@DATASETS.register(name='deve_qa')
class DeVEQADataset(GroundingDataset):
    ANNO_PATH_TRAIN = 'data/deve_qa/DeVE-QA_train.json'
    ANNO_PATH_VALID = 'data/deve_qa/DeVE-QA_val.json'
    ANNO_PATH_VALID_SAMPLE30 = 'data/deve_qa/DeVE-QA_val_sample30.json'

    VIDEO_ROOT = 'data/deve_qa/videos_3fps_480_noaudio'
    DURATIONS = 'data/deve_qa/durations.json'

    @classmethod
    def load_annos(self, split='train', valid_sample30=True,use_qa=False):
        assert use_qa
        if split == 'train':
            raw_annos = nncore.load(self.ANNO_PATH_TRAIN)
        else:
            if valid_sample30:
                raw_annos = nncore.load(self.ANNO_PATH_VALID_SAMPLE30)
            else:
                raw_annos = nncore.load(self.ANNO_PATH_VALID)

        duration_annos = nncore.load(self.DURATIONS)
        annos = []
        for raw_anno in raw_annos:
            question = raw_anno["question"]
            distractors = raw_anno["distract_answers"]
            answer = raw_anno["answer"]
            span = raw_anno["timestamp"]
            vid = raw_anno['vid']
            duration = duration_annos[vid]
            ans_options = distractors.copy()
            ans_options.append(answer)
            random.shuffle(ans_options)
            annos.append(dict(
                source="deve_qa",
                data_type='grounding',
                video_path=nncore.join(self.VIDEO_ROOT, vid + '.mp4'),
                question=question,
                options=ans_options,
                span=[span],
                ans=answer,
                duration=duration,
                query=None
            ))
        return annos
