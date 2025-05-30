import torch
import nncore
from contextlib import nullcontext

from MUPA.constants import GROUNDER_PROMPT, VERIFIER_PROMPT, GQA_PROMPT
from MUPA.dataset.utils import process_vision_info, process_vision_info_v1
from MUPA.utils.parser import parse_query, parse_span, parse_question
from .utils import generate_query, extract_ans, resolve_ans
from MUPA.dataset.hybrid import DATASETS

import random
from collections import Counter
from typing import List, Tuple, Optional, Dict, Any


class GQAAgent:
    def __init__(self, model, processor, args, device):
        self.model = model
        self.processor = processor
        self.args = args
        self.device = device

    def generate_timestamps(self, data, video_ori_fps, indices):
        video_grid_thw = data['video_grid_thw'][0]
        num_frames, window = int(video_grid_thw[0]), int(video_grid_thw[1] * video_grid_thw[2] / 4)
        # 计算时间戳，需要根据temporal_patch_size进行merge
        temporal_patch_size = 2
        time_stamps_insert = [float(idx / video_ori_fps) for idx in indices]  # 计算每帧对应的时间戳（利用帧索引除以帧率）
        timestamps_insert = [round(t, 2) for t in time_stamps_insert]  # 对每个时间戳进行四舍五入，保留两位小数
        use_average = self.args.use_average  # 修改为 False 则采用后一帧的时间戳
        patch_timestamps = []
        num_groups = len(timestamps_insert) // temporal_patch_size  # 30帧合并为15个时间组
        for i in range(num_groups):
            if use_average:
                ts = (timestamps_insert[temporal_patch_size * i] + timestamps_insert[
                    temporal_patch_size * i + 1]) / 2.0
            else:
                ts = timestamps_insert[temporal_patch_size * i + 1]
            patch_timestamps.append(round(ts, 2))  # 保留两位小数
        patch_timestamps = torch.tensor(patch_timestamps)  # [15]
        indices = torch.tensor(indices)
        return num_frames, patch_timestamps, indices

    def run(self, video_path: str, question: str, query: str, duration: float, options: List[str] = None) -> Dict[
        str, Any]:
        print('=============== GQA ===============')
        # build messages
        question = parse_question(question)
        options_prompt = "Options:"
        for idx, opt in enumerate(options):
            letter = chr(ord("A") + idx)
            options_prompt += f"\n({letter}) {opt.capitalize()}"
        prompt_text = GQA_PROMPT.format(question, options_prompt)
        messages = [{
            'role': 'user',
            'content': [{
                'type': 'video',
                'video': video_path,
                'num_threads': self.args.num_threads,
                'min_pixels': 36 * 28 * 28,
                'max_pixels': 64 * 28 * 28,
                'max_frames': 150, # ori 150
                'fps': 1.0
            }, {
                'type': 'text',
                'text': prompt_text
            }]
        }]
        # preprocess and forward
        text = self.processor.apply_chat_template(messages, add_generation_prompt=True)
        if self.args.use_construct:
            images, videos, indices, video_ori_fps = process_vision_info_v1(messages, sanity_check=True)
            data = self.processor(text=[text], images=images, videos=videos, return_tensors='pt')  # Qwen2VLProcessor
            num_frames, patch_timestamps, indices = self.generate_timestamps(data, video_ori_fps, indices)
            # 把采样帧的帧索引列表也加入data
            data["timestamps_insert"] = patch_timestamps
            if self.args.use_duration:
                num_frames = torch.tensor(num_frames)
                data["duration"] = num_frames
        else:
            images, videos = process_vision_info(messages)
            data = self.processor(text=[text], images=images, videos=videos, return_tensors='pt')
        data = data.to(self.device)

        self.model.base_model.disable_adapter_layers()
        self.model.base_model.enable_adapter_layers()
        if 'GQA' in self.model.peft_config:
            self.model.set_adapter('GQA')
        else:
            self.model.set_adapter('grounder')
        output_ids = self.model.generate(
            **data,
            use_construct=self.args.use_construct,
            use_duration=self.args.use_duration,
            use_qa=self.args.use_qa,
            do_sample=False,
            temperature=None,
            top_p=None,
            top_k=None,
            repetition_penalty=None,
            max_new_tokens=256)

        # decode and parse
        assert data.input_ids.size(0) == output_ids.size(0) == 1
        output_ids = output_ids[0, data.input_ids.size(1):]
        if output_ids[-1] == self.processor.tokenizer.eos_token_id:
            output_ids = output_ids[:-1]
        response = self.processor.decode(output_ids, clean_up_tokenization_spaces=False)

        # parse timestamps from model.reg
        ground_success = len(self.model.reg) > 0
        if ground_success:
            # 提取时间戳和置信度
            blob = self.model.reg[0].cpu().float()
            pred, conf = blob[:, :2] * duration, blob[:, -1].tolist()

            # 将预测值限制在 [0, duration] 范围内
            pred = pred.clamp(min=0, max=duration)

            # 根据单位对预测结果进行四舍五入
            unit = 0.001
            pred = torch.round(pred / unit).long() * unit

            # 对异常的预测进行处理
            inds = (pred[:, 1] - pred[:, 0] < 0).nonzero()[:, 0]
            pred[inds] = pred[inds].roll(1)

            # 将预测结果转换为列表
            pred = pred.tolist()
        else:
            print('WARNING: Failed to parse grounder response')
            pred = [[i * duration / 6, (i + 2) * duration / 6] for i in range(5)]
            conf = [0]

        answer = extract_ans(response)

        return {'success': ground_success, 'pred': pred, 'conf': conf, 'answer': answer}


class GrounderAgent:
    def __init__(self, model, processor, args, device):
        self.model = model
        self.processor = processor
        self.args = args
        self.device = device

    def generate_timestamps(self, data, video_ori_fps, indices):
        video_grid_thw = data['video_grid_thw'][0]
        num_frames, window = int(video_grid_thw[0]), int(video_grid_thw[1] * video_grid_thw[2] / 4)
        # 计算时间戳，需要根据temporal_patch_size进行merge
        temporal_patch_size = 2
        time_stamps_insert = [float(idx / video_ori_fps) for idx in indices]  # 计算每帧对应的时间戳（利用帧索引除以帧率）
        timestamps_insert = [round(t, 2) for t in time_stamps_insert]  # 对每个时间戳进行四舍五入，保留两位小数
        use_average = self.args.use_average  # 修改为 False 则采用后一帧的时间戳
        patch_timestamps = []
        num_groups = len(timestamps_insert) // temporal_patch_size  # 30帧合并为15个时间组
        for i in range(num_groups):
            if use_average:
                ts = (timestamps_insert[temporal_patch_size * i] + timestamps_insert[
                    temporal_patch_size * i + 1]) / 2.0
            else:
                ts = timestamps_insert[temporal_patch_size * i + 1]
            patch_timestamps.append(round(ts, 2))  # 保留两位小数
        patch_timestamps = torch.tensor(patch_timestamps)  # [15]
        indices = torch.tensor(indices)
        return num_frames, patch_timestamps, indices

    def run(self, video_path: str, question: str, answer: str, query: str, duration: float) -> Dict[str, Any]:
        print()
        print('=============== grounder ===============')
        # build messages
        if self.args.use_qa and question is not None:  # QA task
            question = parse_question(question)
            if answer is not None:
                ans_letter, ans_text = resolve_ans(answer)
                query = generate_query(question=question, answer=ans_text, use_question=True, use_answer=True)
            else:
                query = generate_query(question=question, answer="", use_question=True, use_answer=False)
        else:  # Moment retrieve task
            assert query is not None
            query=parse_query(query)
        prompt_text = GROUNDER_PROMPT.format(query)
        messages = [{
            'role': 'user',
            'content': [{
                'type': 'video',
                'video': video_path,
                'num_threads': self.args.num_threads,
                'min_pixels': 36 * 28 * 28,
                'max_pixels': 64 * 28 * 28,
                'max_frames': 150,  # origin 150
                'fps': 1.0
            }, {
                'type': 'text',
                'text': prompt_text
            }]
        }]
        # preprocess and forward
        text = self.processor.apply_chat_template(messages, add_generation_prompt=True)
        if self.args.use_construct:
            images, videos, indices, video_ori_fps = process_vision_info_v1(messages, sanity_check=True)
            data = self.processor(text=[text], images=images, videos=videos, return_tensors='pt')  # Qwen2VLProcessor
            num_frames, patch_timestamps, indices = self.generate_timestamps(data, video_ori_fps, indices)
            # 把采样帧的帧索引列表也加入data
            data["timestamps_insert"] = patch_timestamps
            if self.args.use_duration:
                num_frames = torch.tensor(num_frames)
                data["duration"] = num_frames
        else:
            images, videos = process_vision_info(messages)
            data = self.processor(text=[text], images=images, videos=videos, return_tensors='pt')
        data = data.to(self.device)

        self.model.base_model.disable_adapter_layers()
        self.model.base_model.enable_adapter_layers()
        self.model.set_adapter('grounder')
        output_ids = self.model.generate(
            **data,
            use_construct=self.args.use_construct,
            use_duration=self.args.use_duration,
            use_qa=self.args.use_qa,
            do_sample=False,
            temperature=None,
            top_p=None,
            top_k=None,
            repetition_penalty=None,
            max_new_tokens=256)

        # decode and parse
        assert data.input_ids.size(0) == output_ids.size(0) == 1
        output_ids = output_ids[0, data.input_ids.size(1):]
        if output_ids[-1] == self.processor.tokenizer.eos_token_id:
            output_ids = output_ids[:-1]
        response = self.processor.decode(output_ids, clean_up_tokenization_spaces=False)

        # parse timestamps from model.reg
        ground_success = len(self.model.reg) > 0
        if ground_success:
            # 提取时间戳和置信度
            blob = self.model.reg[0].cpu().float()
            pred, conf = blob[:, :2] * duration, blob[:, -1].tolist()

            # 将预测值限制在 [0, duration] 范围内
            pred = pred.clamp(min=0, max=duration)

            # 根据单位对预测结果进行四舍五入
            unit = getattr(DATASETS.get(self.args.dataset), 'UNIT', 0.001)
            pred = torch.round(pred / unit).long() * unit

            # 对异常的预测进行处理
            inds = (pred[:, 1] - pred[:, 0] < 0).nonzero()[:, 0]
            pred[inds] = pred[inds].roll(1)

            # 将预测结果转换为列表
            pred = pred.tolist()
        else:
            pred = [[i * duration / 6, (i + 2) * duration / 6] for i in range(5)]
            conf = [0]

        return {'success': ground_success, 'response': response, 'pred': pred, 'conf': conf}


class VerifierAgent:
    def __init__(self, model, processor, args, device):
        self.model = model
        self.processor = processor
        self.args = args
        self.device = device

    def run(self, video_path: str, question: str, answer: str, query: str, candidates: List[List[float]], conf: List,
            duration: float) -> Dict[str, Any]:
        print('=============== verifier ===============')
        # using top-5 predictions
        # 这里可以选择给verifier全部输入，verifier验证之后poe的时候再top5
        probs = []
        for cand in candidates:
            s0, e0 = parse_span(cand, duration, 2)
            offset = (e0 - s0) / 2
            s1, e1 = parse_span([s0 - offset, e0 + offset], duration)
            s = (s0 - s1) / (e1 - s1)
            e = (e0 - s1) / (e1 - s1)
            if question is not None:  # QA task
                if answer is not None:
                    ans_letter, ans_text = resolve_ans(answer)
                    query = generate_query(question=question, answer=ans_text, use_question=True, use_answer=True)
                else:
                    query = generate_query(question=question, answer="", use_question=True, use_answer=False)
            else:  # Moment retrieve task
                assert query is not None
            prompt_text = VERIFIER_PROMPT.format(query)
            messages = [{
                'role': 'user',
                'content': [{
                    'type': 'video',
                    'video': video_path,
                    'num_threads': self.args.num_threads,
                    'video_start': s1,
                    'video_end': e1,
                    'min_pixels': 36 * 28 * 28,
                    'max_pixels': 64 * 28 * 28,
                    'max_frames': 64,  # origin 64
                    'fps': 2.0
                }, {
                    'type': 'text',
                    'text': prompt_text
                }]
            }]
            text = self.processor.apply_chat_template(messages, add_generation_prompt=True)
            images, videos = process_vision_info(messages)
            data = self.processor(text=[text], images=images, videos=videos, return_tensors='pt')

            # insert seg_start and seg_end
            video_grid_thw = data['video_grid_thw'][0]
            num_frames, window = int(video_grid_thw[0]), int(video_grid_thw[1] * video_grid_thw[2] / 4)
            assert num_frames * window * 4 == data['pixel_values_videos'].size(0)

            pos_s, pos_e = round(s * num_frames), round(e * num_frames)
            pos_s, pos_e = min(max(0, pos_s), num_frames), min(max(0, pos_e), num_frames)
            assert pos_s <= pos_e, (num_frames, s, e)

            base_idx = torch.nonzero(data['input_ids'][0] == self.model.config.vision_start_token_id).item()
            pos_s, pos_e = pos_s * window + base_idx + 1, pos_e * window + base_idx + 2

            input_ids = data['input_ids'][0].tolist()
            input_ids.insert(pos_s, self.model.config.seg_s_token_id)
            input_ids.insert(pos_e, self.model.config.seg_e_token_id)
            data['input_ids'] = torch.LongTensor([input_ids])
            data['attention_mask'] = torch.ones_like(data['input_ids'])
            data = data.to(self.device)

            self.model.base_model.disable_adapter_layers()
            self.model.base_model.enable_adapter_layers()
            self.model.set_adapter('verifier')

            with torch.inference_mode():
                logits = self.model(**data).logits[0, -1].softmax(dim=-1)

            # NOTE: magic numbers here
            # In Qwen2-VL vocab: 9454 -> Yes, 2753 -> No
            score = (logits[9454] - logits[2753]).sigmoid().item()
            probs.append(score)

        ranks = torch.Tensor(probs).argsort(descending=True).tolist()  # 根据得分降序排序候选预测
        pred = [candidates[idx] for idx in ranks]
        conf = [probs[idx] for idx in ranks]

        return {'probs': probs, 'ranks': ranks, 'pred': pred, 'conf': conf}


class AnswererAgent:
    def __init__(self, model, processor, args, device, adapter_state):
        self.model = model
        self.processor = processor
        self.args = args
        self.device = device
        self.adapter_state = adapter_state

    def run(self, video_path: str, question: str, selected_span: List[float], duration: float,
            options: List[str] = None) -> str:
        print('=============== answerer ===============')
        min_len = getattr(DATASETS.get(self.args.dataset), 'MIN_LEN', 32)
        start, end = parse_span(selected_span, duration, min_len)
        assert question is not None
        if self.args.style in ('mcq', 'options'):  # 多选题 mcq=Multiple Choice Question
            prompt_text = question + '\nOptions:'
            for idx, opt in enumerate(options):
                prompt_text += f"\n({chr(ord('A') + idx)}) {opt.capitalize()}"
            prompt_text += '\nPlease only give the best option.'
        else:
            prompt_text = question
        messages = [{
            'role': 'user',
            'content': [{
                'type': 'video',
                'video': video_path,
                'num_threads': self.args.num_threads,
                'video_start': start,
                'video_end': end,
                'min_pixels': 128 * 28 * 28,
                'max_pixels': 256 * 28 * 28,
                'max_frames': 32, # ori 32
                'fps': 2.0
            }, {
                'type': 'text',
                'text': prompt_text
            }]
        }]
        text = self.processor.apply_chat_template(messages, add_generation_prompt=True)
        images, videos = process_vision_info(messages)
        data = self.processor(text=[text], images=images, videos=videos, return_tensors='pt').to(self.device)
        # set adapter
        if self.adapter_state:
            self.model.base_model.disable_adapter_layers()
            self.model.base_model.enable_adapter_layers()
            self.model.set_adapter('answerer')
            context = nullcontext
        else:
            context = self.model.disable_adapter
        with context():
            output_ids = self.model.generate(
                **data,
                do_sample=False,
                temperature=None,
                top_p=None,
                top_k=None,
                repetition_penalty=None,
                max_new_tokens=256)
        assert data.input_ids.size(0) == output_ids.size(0) == 1
        output_ids = output_ids[0, data.input_ids.size(1):]
        if output_ids[-1] == self.processor.tokenizer.eos_token_id:
            output_ids = output_ids[:-1]
        response = self.processor.decode(output_ids, clean_up_tokenization_spaces=False)

        return {'answer': response}


class ReflectionAgent:
    def __init__(self, verifier: VerifierAgent, n_clusters=5, kmeans_iters=10):
        self.verify = verifier
        self.n_clusters = n_clusters
        self.kmeans_iters = kmeans_iters

    # --- PoE for spans ---
    def poe_span(self,
                 spans: List[Tuple[float, float]],
                 pred_confs: List[float],
                 ver_confs: List[float],
                 top_k: int = 5
                 ) -> List[Tuple[Tuple[float, float], float]]:
        combined = [(s, p * v) for s, p, v in zip(spans, pred_confs, ver_confs)]
        combined.sort(key=lambda x: x[1], reverse=True)
        return combined[:top_k]

    # --- PoE for answers ---
    def poe_answer(self, answers: List[str]) -> str:
        parsed: List[Tuple[str, str]] = [(resolve_ans(answer)[0],answer) for answer in answers]
        letters = [letter for letter, _ in parsed]
        cnt = Counter(letters)
        top_letter, top_count = cnt.most_common(1)[0]
        if top_count > 1:
            for letter, answer in parsed:
                if letter == top_letter:
                    return answer
        return random.choice(answers)

    def weighted_kmeans(self,
                        spans: List[Tuple[float, float]],
                        confs: List[float],
                        n_clusters: int = 5,
                        max_iters: int = 10,
                        eps: float = 1e-6) -> Tuple[List[Tuple[float, float]], List[int]]:
        """
        Weighted K-means clustering with confidence-based weighting.
        - spans: List of span tuples [(start, end)].
        - confs: List of confidence values for each span.
        - n_clusters: Number of clusters to form.
        - max_iters: Maximum number of iterations to perform.
        - eps: A small value to avoid division by zero.
        """
        # Step 1: Initialize centers with the top n_clusters spans based on the highest confidence.
        idxs = sorted(range(len(spans)), key=lambda i: confs[i], reverse=True)[:n_clusters]
        centers = [spans[i] for i in idxs]
        labels = [0] * len(spans)

        for _ in range(max_iters):
            # Step 2: Assign each span to the nearest center
            for j, s in enumerate(spans):
                dists = [(s[0] - c[0]) ** 2 + (s[1] - c[1]) ** 2 for c in centers]
                labels[j] = int(min(range(n_clusters), key=lambda k: dists[k]))

            # Step 3: Update centers based on the members' weighted averages
            new_centers = []
            for k in range(n_clusters):
                members = [i for i, lbl in enumerate(labels) if lbl == k]
                if not members:
                    # If no members are assigned to this cluster, keep the original center
                    new_centers.append(centers[k])
                else:
                    # Calculate the total weight (confidence sum) for the current cluster
                    total_w = sum(confs[i] for i in members) + eps  # Prevent division by zero

                    # Calculate the weighted average of spans (start and end)
                    s_avg = sum(spans[i][0] * confs[i] for i in members) / total_w
                    e_avg = sum(spans[i][1] * confs[i] for i in members) / total_w
                    new_centers.append((s_avg, e_avg))

            # Step 4: Update centers for the next iteration
            centers = new_centers

        return centers, labels

    # --- MoE for spans via weighted K-means ---
    def moe_span(self,
                 spans: List[Tuple[float, float]],
                 confs: List[float]
                 ) -> List[Tuple[Tuple[float, float], float]]:
        centers, labels = self.weighted_kmeans(spans, confs, n_clusters=self.n_clusters, max_iters=self.kmeans_iters)
        cluster_confs = []
        for k in range(self.n_clusters):
            members = [i for i, lbl in enumerate(labels) if lbl == k]
            if not members:
                cluster_confs.append(0.0)
            else:
                avg_w = sum(confs[i] for i in members) / len(members)
                cluster_confs.append(avg_w)
        results = list(zip(centers, cluster_confs))
        results.sort(key=lambda x: x[1], reverse=True)
        return results

    # --- 三种功能接口 ---
    def single_path_with_verifier(self, video_path, question, answer, query, pred_spans, pred_confs, duration,
                                  top_k=5) -> Dict:
        # 1) 调用 Verifier 重新评分排序
        verifier_out = self.verify.run(video_path=video_path, question=question, answer=answer, query=query,
                                       candidates=pred_spans, conf=pred_confs, duration=duration)
        ver_spans = verifier_out['pred']
        ver_confs = verifier_out['conf']
        ver_scores = verifier_out['probs']
        # 2) PoE 选 Top-k
        fused = self.poe_span(pred_spans, pred_confs, ver_scores, top_k)
        reflection_spans, reflection_confs = zip(*fused)
        return dict(verifier_spans=ver_spans, verifier_confs=ver_confs, reflection_spans=list(reflection_spans),
                    reflection_confs=list(reflection_confs))

    def multi_path_answer_fusion(self, answers: List[str]) -> str:
        return self.poe_answer(answers)

    def multi_path_span_fusion(self, all_spans, all_confs):
        fused = self.moe_span(all_spans, all_confs)
        reflection_spans, reflection_confs = zip(*fused)
        return dict(reflection_spans=reflection_spans, reflection_confs=reflection_confs)

    # --- 统一入口 ---
    def run(self, mode: str, *args, **kwargs) -> Any:
        if mode == 'verify':
            return self.single_path_with_verifier(*args, **kwargs)
        elif mode == 'reflect_ans':
            return self.multi_path_answer_fusion(*args, **kwargs)
        elif mode == 'reflect_span':
            return self.multi_path_span_fusion(*args, **kwargs)
        else:
            raise ValueError(f"Unknown mode {mode}")

# Example usage:
# grounder = GrounderAgent(model, processor, args, device)
# ground_res = grounder.run(video_path, query, duration, options, answer)
# verifier = VerifierAgent(model, processor, args, device)
# ver_res = verifier.run(video_path, query, ground_res['pred'], duration)
# answerer = AnswererAgent(model, processor, args, device)
# final_answer = answerer.run(video_path, question, ground_res['pred'][ver_res['best_idx']], duration)
