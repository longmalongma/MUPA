# Copyright (c) 2025 Huilin Song. Licensed under the BSD-3-Clause License.

import argparse
import copy
import nncore

from MUPA.dataset.hybrid import DATASETS
from MUPA.model.builder import build_model, build_model_multipath
from MUPA.utils.io import get_duration
from MUPA.model.multi_agent import GQAAgent, GrounderAgent, VerifierAgent, AnswererAgent, ReflectionAgent, \
    MultiPathInference_opt


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset')
    parser.add_argument('--pred_path')
    parser.add_argument('--model_path')
    parser.add_argument('--split', default='test', choices=['train', 'valid', 'test'])
    parser.add_argument('--style', default='mcq', choices=['mcq', 'options', 'direct'])
    parser.add_argument('--use_subtitle', action='store_true')
    parser.add_argument('--auto_rephrasing', action='store_true')
    parser.add_argument('--auto_planning', action='store_true')
    parser.add_argument('--num_threads', type=int, default=1)
    parser.add_argument('--device', default='auto')
    parser.add_argument('--chunk', type=int, default=1)
    parser.add_argument('--index', type=int, default=0)
    parser.add_argument('--use_construct', type=bool, default=True)
    parser.add_argument('--use_duration', type=bool, default=True)
    parser.add_argument('--use_average', type=bool, default=True)
    parser.add_argument('--use_qa', type=bool, default=False)
    parser.add_argument('--paths', type=str, default="path1,path2,path3")
    parser.add_argument('--reflect', type=bool)
    parser.add_argument('--task', type=str, choices=['GQA', 'MR'])
    args = parser.parse_args()
    return args


if __name__ == '__main__':
    args = parse_args()
    assert args.task == "GQA" or args.task == "MR", "Unknown task type"
    if args.task == "GQA":
        assert args.reflect, "In GQA task, relfect must be true"

    if args.chunk > 1:
        pred_path = nncore.join(args.pred_path, f'output_{args.index}.json')
    else:
        pred_path = nncore.join(args.pred_path, 'output.json')

    print(f'Dataset: {args.dataset}({args.split}) Chunk: {args.chunk} Index: {args.index} Output Path: {pred_path}')

    model, processor = build_model_multipath(model_path=args.model_path, device=args.device)
    device = next(model.parameters()).device

    gqa_agent = GQAAgent(model, processor, args, device)
    grounder_agent = GrounderAgent(model, processor, args, device)
    verifier_agent = VerifierAgent(model, processor, args, device)
    answerer_agent = AnswererAgent(model, processor, args, device, adapter_state=False)
    reflection_agent = ReflectionAgent(verifier=verifier_agent, n_clusters=5, kmeans_iters=10)
    multi_path = MultiPathInference_opt(gqa_agent=gqa_agent,
                                        grounder_agent=grounder_agent,
                                        verifier_agent=verifier_agent,
                                        answerer_agent=answerer_agent,
                                        reflection_agent=reflection_agent)
    paths = args.paths.split(',')
    annos = DATASETS.get(args.dataset).load_annos(split=args.split, use_qa=args.use_qa)
    annos = [annos[i::args.chunk] for i in range(args.chunk)][args.index]

    dumps = []
    for i in nncore.ProgressBar(range(len(annos))):
        anno = copy.deepcopy(annos[i])
        dump = copy.deepcopy(annos[i])

        video_path, duration, span = anno['video_path'].replace("\\", "/"), anno.get('duration'), anno.get('span')

        if duration is None:
            duration = get_duration(video_path, num_threads=args.num_threads)
            dump['duration'] = duration

        question = anno.get('question', None)
        options = anno.get('options', None)
        ans = anno.get('ans', None)
        query = anno['query']

        multi_path_ans = []
        all_ground_spans = []
        all_ground_confs = []
        all_ver_spans = []
        all_ver_confs = []
        all_reflection_spans = []
        all_reflection_confs = []
        if "path1" in paths:
            path1_result = multi_path.run(mode='path1',
                                          video_path=video_path,
                                          question=question,
                                          query=query,
                                          duration=duration,
                                          options=options)
            path1_answer = path1_result["answer"]
            path1_grounder_spans = path1_result["grounder_spans"]
            path1_grounder_confs = path1_result["grounder_confs"]
            path1_ver_spans = path1_result["verifier_spans"]
            path1_ver_confs = path1_result["verifier_confs"]
            path1_reflection_spans = path1_result["reflection_spans"]
            path1_reflection_confs = path1_result["reflection_confs"]
            # dump['path1'] = dict(answer=path1_answer,
            #                      ground=dict(pred_span=path1_grounder_spans, conf=path1_grounder_confs),
            #                      verify=dict(pred_span=path1_ver_spans, conf=path1_ver_confs),
            #                      reflection=dict(pred_span=path1_reflection_spans, conf=path1_reflection_confs))
            multi_path_ans.append(path1_answer)
            all_ground_spans += path1_grounder_spans
            all_ground_confs += path1_grounder_confs
            all_ver_spans += path1_ver_spans
            all_ver_confs += path1_ver_confs
            all_reflection_spans += path1_reflection_spans
            all_reflection_confs += path1_reflection_confs

        if "path2" in paths:
            path2_result = multi_path.run(mode='path2',
                                          video_path=video_path,
                                          question=question,
                                          query=query,
                                          duration=duration,
                                          options=options)
            path2_answer = path2_result["answer"]
            path2_gqa_spans = path2_result["gqa_spans"]
            path2_gqa_confs = path2_result["gqa_confs"]
            path2_ver_spans = path2_result["verifier_spans"]
            path2_ver_confs = path2_result["verifier_confs"]
            path2_reflection_spans = path2_result["reflection_spans"]
            path2_reflection_confs = path2_result["reflection_confs"]
            # dump['path2'] = dict(answer=path2_answer,
            #                      ground=dict(pred_span=path2_gqa_spans, conf=path2_gqa_confs),
            #                      verify=dict(pred_span=path2_ver_spans, conf=path2_ver_confs),
            #                      reflection=dict(pred_span=path2_reflection_spans, conf=path2_reflection_confs))
            multi_path_ans.append(path2_answer)
            all_ground_spans += path2_gqa_spans
            all_ground_confs += path2_gqa_confs
            all_ver_spans += path2_ver_spans
            all_ver_confs += path2_ver_confs
            all_reflection_spans += path2_reflection_spans
            all_reflection_confs += path2_reflection_confs

        if "path3" in paths:
            path3_result = multi_path.run(mode='path3',
                                          video_path=video_path,
                                          question=question,
                                          query=query,
                                          duration=duration,
                                          options=options)
            path3_answer = path3_result["answer"]
            path3_grounder_spans = path3_result["grounder_spans"]
            path3_grounder_confs = path3_result["grounder_confs"]
            path3_ver_spans = path3_result["verifier_spans"]
            path3_ver_confs = path3_result["verifier_confs"]
            path3_reflection_spans = path3_result["reflection_spans"]
            path3_reflection_confs = path3_result["reflection_confs"]
            # dump['path3'] = dict(answer=path3_answer,
            #                      ground=dict(pred_span=path3_grounder_spans, conf=path3_grounder_confs),
            #                      verify=dict(pred_span=path3_ver_spans, conf=path3_ver_confs),
            #                      reflection=dict(pred_span=path3_reflection_spans, conf=path3_reflection_confs))
            multi_path_ans.append(path3_answer)
            all_ground_spans += path3_grounder_spans
            all_ground_confs += path3_grounder_confs
            all_ver_spans += path3_ver_spans
            all_ver_confs += path3_ver_confs
            all_reflection_spans += path3_reflection_spans
            all_reflection_confs += path3_reflection_confs

        if args.task == "GQA" and args.reflect:
            # answer reflection
            reflection_answer = reflection_agent.run(mode="reflect_ans", answers=multi_path_ans)


            def sort_spans(spans, confs):
                paired = list(zip(spans, confs))
                paired.sort(key=lambda x: x[1], reverse=True)
                sorted_spans, sorted_confs = zip(*paired) if paired else ([], [])
                return sorted_spans, sorted_confs


            all_ground_spans, all_ground_confs = sort_spans(all_ground_spans, all_ground_confs)
            all_ver_spans, all_ver_confs = sort_spans(all_ver_spans, all_ver_confs)
            all_reflection_spans, all_reflection_confs = sort_spans(all_reflection_spans, all_reflection_confs)
            # span reflection
            ground_reflection_out = reflection_agent.run(mode="reflect_span", all_spans=all_ground_spans,
                                                         all_confs=all_ground_confs)
            ground_reflection_spans = ground_reflection_out["reflection_spans"]
            ground_reflection_confs = ground_reflection_out["reflection_confs"]

            ver_reflection_out = reflection_agent.run(mode="reflect_span", all_spans=all_ver_spans,
                                                      all_confs=all_ver_confs)
            ver_reflection_spans = ver_reflection_out["reflection_spans"]
            ver_reflection_confs = ver_reflection_out["reflection_confs"]

            final_reflection_out = reflection_agent.run(mode="reflect_span", all_spans=all_reflection_spans,
                                                        all_confs=all_reflection_confs)
            final_reflection_spans = final_reflection_out["reflection_spans"]
            final_reflection_confs = final_reflection_out["reflection_confs"]

            # dump["multipath"] = dict(answer=reflection_answer,
            #                          ground=dict(pred_span=ground_reflection_spans, conf=ground_reflection_confs),
            #                          verify=dict(pred_span=ver_reflection_spans, conf=ver_reflection_confs),
            #                          reflection=dict(pred_span=final_reflection_spans, conf=final_reflection_confs))

            dump["pred_span"] = reflection_answer
            dump["pred_span"] = final_reflection_spans[:5]  # top-5
            dump["span_conf"] = final_reflection_confs[:5]

        if "path4" in paths and args.task == "MR":
            path4_result = multi_path.run(mode='path4',
                                          video_path=video_path,
                                          question=question,
                                          query=query,
                                          duration=duration)

            path4_grounder_spans = path4_result["grounder_spans"]
            path4_grounder_confs = path4_result["grounder_confs"]
            path4_ver_spans = path4_result["verifier_spans"]
            path4_ver_confs = path4_result["verifier_confs"]
            dump['path4'] = dict(ground=dict(pred_span=path4_grounder_spans, conf=path4_grounder_confs),
                                 verify=dict(pred_span=path4_ver_spans, conf=path4_ver_confs))

            dump["pred_span"] = path4_ver_spans[:5]  # top-5
            dump["span_conf"] = path4_ver_confs[:5]

        dumps.append(dump)

    nncore.dump(dumps, pred_path)
