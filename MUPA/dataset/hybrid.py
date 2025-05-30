# Copyright (c) Huilin Song. Licensed under the BSD 3-Clause License.

import math
import random
from collections import defaultdict
from itertools import accumulate

import nncore
import numpy as np
import termplotlib as tpl
import torch
from tabulate import tabulate
from torch.utils.data import Dataset

from MUPA.constants import IGNORE_INDEX
from MUPA.dataset.utils import preprocess, process_vision_info, process_vision_info_v1
from MUPA.utils.parser import parse_span

DATASETS = nncore.Registry('datasets')


class HybridDataset(Dataset):

    def __init__(self, processor, model_config, model_args, data_args, training_args):
        super().__init__()

        datasets = []
        for key in data_args.datasets.split(','):
            datasets.append(DATASETS.get(key)(processor, model_args, data_args, training_args))  # 调用各个数据集的构造函数

        data_types = [a['data_type'] for d in datasets for a in d.annos]

        cum_length = [0] + list(accumulate([len(d) for d in datasets]))
        idx_ranges = [[cum_length[i], cum_length[i + 1]] for i in range(len(cum_length) - 1)]

        if training_args.local_rank in (0, -1):
            raw_length = sum(d.raw_length for d in datasets)
            cur_length = idx_ranges[-1][-1]

            ratio = round(cur_length / raw_length * 100, 2)
            print(f'Number of samples: {raw_length} (original) -> {cur_length} (filtered) {ratio}%')

            data_type_cnt = ' '.join([f'{data_types.count(t)} ({t})' for t in list(set(data_types))])
            print(f'Data types: {data_type_cnt}')

            tab = defaultdict(int)
            for dataset in datasets:
                for anno in dataset.annos:
                    tab[anno.get('source', 'unknown')] += 1

            tab = [[k, v, round(v / cur_length, 3)] for k, v in tab.items()]
            print(tabulate(tab, headers=['Source', '#Samples', 'Ratio'], tablefmt='pretty', stralign='left'))

            d, _ = torch.Tensor([a['duration'] for d in datasets for a in d.annos if 'duration' in a]).sort()
            if d.size(0) > 0:
                n, r = min(d.size(0), 10), d.flip(0)
                print(f'Top-{n} max video durations: {[round(r[i].item(), 1) for i in range(n)]}')
                print(f'Top-{n} min video durations: {[round(d[i].item(), 1) for i in range(n)]}')
                print(f'Average video duration ({d.size(0)} samples): {round(d.mean().item(), 1)}s')

                print('Video duration histogram:')
                counts, edges = np.histogram(d)
                labels = [f'{edges[i]:.2f}s - {edges[i + 1]:.2f}s' for i in range(len(edges) - 1)]
                fig = tpl.figure()
                fig.barh(counts, labels)
                fig.show()

            d, _ = torch.Tensor([abs(b[0] - b[1]) for d in datasets for a in d.annos if 'span' in a
                                 for b in a['span']]).sort()
            if d.size(0) > 0:
                n, r = min(d.size(0), 10), d.flip(0)
                print(f'Top-{n} max span durations: {[round(r[i].item(), 1) for i in range(n)]}')
                print(f'Top-{n} min span durations: {[round(d[i].item(), 1) for i in range(n)]}')
                print(f'Average span duration ({d.size(0)} samples): {round(d.mean().item(), 1)}s')

                print('Span duration histogram:')
                counts, edges = np.histogram(d)
                labels = [f'{edges[i]:.2f}s - {edges[i + 1]:.2f}s' for i in range(len(edges) - 1)]
                fig = tpl.figure()
                fig.barh(counts, labels)
                fig.show()

        self.datasets = datasets
        self.data_types = data_types
        self.idx_ranges = idx_ranges
        self.processor = processor
        self.model_config = model_config
        self.model_args = model_args
        self.data_args = data_args
        self.training_args = training_args

    def __len__(self):
        return self.idx_ranges[-1][-1]

    def __getitem__(self, idx):
        for retry in range(self.data_args.max_retries + 1):
            try:
                if self.model_args.use_construct:
                    return self.fetch_data_v1(idx)
                else:
                    return self.fetch_data(idx)
            except Exception as e:
                print(f'Error in loading {idx}: {type(e).__name__}({e})')
                idx = random.choice([i for i, t in enumerate(self.data_types) if t == self.data_types[idx]])

        raise RuntimeError(f'Data loading failed after {retry} retries')

    def map(self, *args, **kwargs):
        return self

    def fetch_data(self, idx):
        for (s, e), dataset in zip(self.idx_ranges, self.datasets):
            if s <= idx < e:
                meta = dataset[idx - s]
                break

        text = self.processor.apply_chat_template(meta['messages'])
        text = [text.strip()]

        images, videos = process_vision_info(meta['messages'], sanity_check=True)

        data = self.processor(text=text, images=images, videos=videos,return_tensors='pt')
        assert data['input_ids'].size(0) == 1

        data['input_ids'] = data['input_ids'][0]  # 降维
        data['labels'] = preprocess(data['input_ids'], text[0], self.processor.tokenizer, self.model_args.conv_type)

        # insert segment start/end tokens
        if 'ss' in meta and 'se' in meta:
            video_grid_thw = data['video_grid_thw'][0]
            num_frames, window = int(video_grid_thw[0]), int(video_grid_thw[1] * video_grid_thw[2] / 4)
            assert num_frames * window * 4 == data['pixel_values_videos'].size(0)

            pos_s, pos_e = round(meta['ss'] * num_frames), round(meta['se'] * num_frames)
            pos_s, pos_e = min(max(0, pos_s), num_frames), min(max(0, pos_e), num_frames)
            assert pos_s <= pos_e, (num_frames, meta['ss'], meta['se'])

            base_idx = torch.nonzero(data['input_ids'] == self.model_config.vision_start_token_id).item()
            pos_s, pos_e = pos_s * window + base_idx + 1, pos_e * window + base_idx + 2

            input_ids = data['input_ids'].tolist()
            input_ids.insert(pos_s, self.model_config.seg_s_token_id)
            input_ids.insert(pos_e, self.model_config.seg_e_token_id)
            data['input_ids'] = torch.LongTensor(input_ids)

            labels = data['labels'].tolist()
            labels.insert(pos_s, IGNORE_INDEX)
            labels.insert(pos_e, IGNORE_INDEX)
            data['labels'] = torch.LongTensor(labels)

        if 'span' in meta:
            span, duration = meta['span'], meta['duration']

            pixel_values_videos, video_grid_thw = data['pixel_values_videos'], data['video_grid_thw']
            num_frames = int(video_grid_thw[0][0])

            assert video_grid_thw.size(0) == 1
            assert video_grid_thw.prod() == pixel_values_videos.size(0)

            # actual fps would be 1/2 of config (temporal patch size = 2)
            fps = num_frames / duration

            safe_span = [parse_span(b, duration, 1 / fps) for b in span]

            # num_reg_tokens -> num_bnds -> s & e
            timestamps = [[[s / duration, e / duration] for s, e in safe_span]]

            saliency, pos_inds = torch.zeros(num_frames), []
            for s, e in safe_span:
                span_ind = max(0, s * fps), min(e * fps, num_frames)
                pos_inds = list(range(math.ceil(span_ind[0]), math.ceil(span_ind[1])))
                assert len(pos_inds) > 0, f'empty pos_inds ({idx}): {fps} {num_frames} {duration} {span}'
                saliency[pos_inds] = 1

            assert saliency.any(), f'empty saliency ({idx}): {pos_inds} {fps} {num_frames} {duration} {span}'
            pos_clip = random.sample(saliency.nonzero()[:, 0].tolist(), 1)
            pos_clip = torch.LongTensor(pos_clip)

            data['timestamps'] = timestamps
            data['saliency'] = saliency
            data['pos_clip'] = pos_clip

        return data

    def fetch_data_v1(self, idx):
        """
        Retrieve and preprocess a data sample by global index:
        - Locate the appropriate dataset and metadata
        - Format text input with chat template
        - Extract vision features and tokenize
        - Optionally insert segment tokens and compute grounding information
        """
        # Locate metadata for the given index
        for (start, end), dataset in zip(self.idx_ranges, self.datasets):
            if start <= idx < end:
                meta = dataset[idx - start]
                break

        # Prepare text input from messages
        text = self.processor.apply_chat_template(meta['messages']).strip()
        text = [text]

        # Process vision data: returns images, videos, frame indices, and original FPS
        images, videos, indices, video_ori_fps = process_vision_info_v1(
            meta['messages'], sanity_check=True
        )

        # Tokenize inputs and generate labels
        data = self.processor(
            text=text, images=images, videos=videos, return_tensors='pt'
        )
        assert data['input_ids'].size(0) == 1
        data['input_ids'] = data['input_ids'][0]
        data['labels'] = preprocess(
            data['input_ids'], text[0], self.processor.tokenizer, self.model_args.conv_type
        )

        # Insert segment start/end tokens if defined in metadata
        if 'ss' in meta and 'se' in meta:
            grid = data['video_grid_thw'][0]
            num_frames = int(grid[0])
            window_size = int(grid[1] * grid[2] / 4)
            assert num_frames * window_size * 4 == data['pixel_values_videos'].size(0)

            # Compute token insertion positions based on temporal spans
            pos_start = round(meta['ss'] * num_frames)
            pos_end = round(meta['se'] * num_frames)
            pos_start = min(max(0, pos_start), num_frames)
            pos_end = min(max(0, pos_end), num_frames)
            assert pos_start <= pos_end, (num_frames, meta['ss'], meta['se'])

            # Find the index of the vision start token
            base_idx = torch.nonzero(
                data['input_ids'] == self.model_config.vision_start_token_id
            ).item()
            pos_start = pos_start * window_size + base_idx + 1
            pos_end = pos_end * window_size + base_idx + 2

            # Insert segment tokens into input_ids and labels
            input_ids = data['input_ids'].tolist()
            input_ids.insert(pos_start, self.model_config.seg_s_token_id)
            input_ids.insert(pos_end, self.model_config.seg_e_token_id)
            data['input_ids'] = torch.LongTensor(input_ids)

            labels = data['labels'].tolist()
            labels.insert(pos_start, IGNORE_INDEX)
            labels.insert(pos_end, IGNORE_INDEX)
            data['labels'] = torch.LongTensor(labels)

        # Compute grounding and saliency information if span is provided
        if 'span' in meta:
            span, duration = meta['span'], meta['duration']
            pixel_values = data['pixel_values_videos']
            grid_info = data['video_grid_thw']

            num_frames = int(grid_info[0][0])
            assert grid_info.size(0) == 1
            assert grid_info.prod() == pixel_values.size(0)

            # Derive frames-per-second for patch-level frames
            fps = num_frames / duration
            safe_spans = [parse_span(b, duration, 1 / fps) for b in span]

            # Normalize timestamps for each span
            timestamps = [[[s / duration, e / duration] for s, e in safe_spans]]

            # Build saliency mask and sample a positive clip
            saliency = torch.zeros(num_frames)
            pos_indices = []
            for s, e in safe_spans:
                start_frame = max(0, s * fps)
                end_frame = min(e * fps, num_frames)
                idx_range = list(range(math.ceil(start_frame), math.ceil(end_frame)))
                assert idx_range, f"Empty index range for sample {idx}"
                saliency[idx_range] = 1
                pos_indices = idx_range

            assert saliency.any(), f"No salient frames for sample {idx}"
            sampled_clip = random.sample(saliency.nonzero()[:, 0].tolist(), 1)
            data['timestamps'] = timestamps
            data['saliency'] = saliency
            data['pos_clip'] = torch.LongTensor(sampled_clip)

            # Optionally include temporal patch timestamps for construct mode
            if self.model_args.use_construct:
                grid = data['video_grid_thw'][0]
                num_frames = int(grid[0])
                window_size = int(grid[1] * grid[2] / 4)
                patch_size = 2  # defined in config for Qwen2-VL-2B-Instruct

                # Compute raw timestamps for each sampled frame
                raw_times = [idx / video_ori_fps for idx in indices]
                rounded_times = [round(t, 2) for t in raw_times]

                # Merge frames into patches
                patch_timestamps = []
                groups = len(rounded_times) // patch_size
                for i in range(groups):
                    t1 = rounded_times[patch_size * i]
                    t2 = rounded_times[patch_size * i + 1]
                    if self.model_args.use_average:
                        patch_ts = (t1 + t2) / 2.0
                    else:
                        patch_ts = t2
                    patch_timestamps.append(round(patch_ts, 2))

                data['timestamps_insert'] = torch.tensor(patch_timestamps)
                data['use_construct'] = True
                if self.model_args.use_duration:
                    data['use_duration'] = True
                    data['duration'] = torch.tensor(num_frames)

        return data

