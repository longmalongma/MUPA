import torch
import argparse

import nncore
from nncore.ops import temporal_area, temporal_intersection, temporal_iof, temporal_iou
from tabulate import tabulate
import re

class SafeInt(int):

    def __truediv__(self, other):
        try:
            return SafeInt(super().__truediv__(other))
        except ZeroDivisionError:
            return SafeInt(0)


def pure_response(response):
    pattern = re.compile(
        r"""
        (?:Answer:\s*)?         # optional 'Answer:' prefix
        \(                      # opening parenthesis
          ([A-Z])                # capture a letter A-Z
        \)                      # closing parenthesis
        \s*                     # optional whitespace
        (.+?)                   # capture answer text
        $                        # end of string
        """,
        re.VERBOSE
    )
    m = pattern.search(response.strip())
    if m:
        letter = m.group(1)
        text = m.group(2).strip()
        full = f"({letter}) {text}"
    else:
        full = response.strip()
        m2 = re.search(r"\(([A-Z])\)", full)
        letter = m2.group(1) if m2 else None
        text = full.split(")", 1)[-1].strip()
    return full

def check_ans(options, ans, response):
    a = ans.lower()
    response = pure_response(response)
    b = response.lower().split(' ')[0].replace('(', '').replace(')', '').replace('.', '')
    if len(b) != 1:
        b = b[0]
        nncore.log(f'WARNING: {response} -> {b}')
    if b not in [chr(ord('a') + i) for i in range(len(options))]:
        nncore.log(f'ERROR: {response} -> {b}')
        return
    return a == b


def compute_iou_convex_hull(pred, span, conf, cgbench_mode, conf_thr):
    pred_tensor = torch.Tensor(pred)
    span_tensor = torch.Tensor(span)

    if cgbench_mode:
        if conf_thr > 0:
            conf_tensor = torch.Tensor(conf)
            keep = torch.cat((
                torch.LongTensor([0]),
                torch.where(conf_tensor > conf_thr)[0]
            )).unique()
            pred_tensor = pred_tensor[keep]
        else:
            pred_tensor = pred_tensor[:1]

        pred_area = temporal_area(pred_tensor).sum()
        span_area = temporal_area(span_tensor).sum()
        inter = temporal_intersection(pred_tensor, span_tensor).sum()
        iou = (inter / (pred_area + span_area - inter)).unsqueeze(0)
        return torch.where(iou.isfinite(), iou, torch.zeros_like(iou))

    if span_tensor.size(0) == 1:
        iou = temporal_iou(pred_tensor, span_tensor).squeeze(1)
        return torch.where(iou.isfinite(), iou, torch.zeros_like(iou))

    starts = span_tensor[:, 0]
    ends = span_tensor[:, 1]
    hull = torch.tensor([[starts.min(), ends.max()]])  # shape (1,2)

    iou = temporal_iou(pred_tensor, hull).squeeze(1)
    return torch.where(iou.isfinite(), iou, torch.zeros_like(iou))


def compute_iou_multi_interval(pred, span, conf, cgbench_mode, conf_thr):
    pred_tensor = torch.Tensor(pred)
    span_tensor = torch.Tensor(span)

    if cgbench_mode:
        if conf_thr > 0:
            conf_tensor = torch.Tensor(conf)
            keep = torch.cat((
                torch.LongTensor([0]),
                torch.where(conf_tensor > conf_thr)[0]
            )).unique()
            pred_tensor = pred_tensor[keep]
        else:
            pred_tensor = pred_tensor[:1]

        pred_area = temporal_area(pred_tensor).sum()
        span_area = temporal_area(span_tensor).sum()
        inter = temporal_intersection(pred_tensor, span_tensor).sum()
        iou = (inter / (pred_area + span_area - inter)).unsqueeze(0)
        return torch.where(iou.isfinite(), iou, torch.zeros_like(iou))

    if span_tensor.size(0) == 1:
        iou = temporal_iou(pred_tensor, span_tensor).squeeze(1)
        return torch.where(iou.isfinite(), iou, torch.zeros_like(iou))

    inter_mat = temporal_intersection(pred_tensor, span_tensor)  # (N_pred, N_gt)
    inter_sum = inter_mat.sum(dim=1)  # (N_pred,)

    area_pred = temporal_area(pred_tensor)  # (N_pred,)
    area_gt_sum = temporal_area(span_tensor).sum()  # scalar

    denom = area_pred + area_gt_sum - inter_sum  # (N_pred,)
    iou = torch.where(denom > 0, inter_sum / denom, torch.zeros_like(inter_sum))
    return iou


def compute_iou(pred, span, conf, cgbench_mode, conf_thr):
    pred_tensor = torch.Tensor(pred)
    span_tensor = torch.Tensor(span)

    if cgbench_mode:
        if conf_thr > 0:
            conf_tensor = torch.Tensor(conf)
            keep = torch.cat((torch.LongTensor([0]), torch.where(conf_tensor > conf_thr)[0])).unique()
            pred_tensor = pred_tensor[keep]
        else:
            pred_tensor = pred_tensor[:1]
        pred_area = temporal_area(pred_tensor).sum()
        span_area = temporal_area(span_tensor).sum()
        inter = temporal_intersection(pred_tensor, span_tensor).sum()
        iou = (inter / (pred_area + span_area - inter)).unsqueeze(0)
        assert iou.numel() == 1
    else:
        iou = temporal_iou(pred_tensor, span_tensor)

    iou = torch.where(iou.isfinite(), iou, 0)
    return iou


def init_counters(top_k, thres):
    return {
        'tab_iou': {},  # dict: task -> list counters
        'tab_iop': {},
        'tab_ans': {},
        'tab_iou_all': [0] * (len(top_k) * len(thres) + 3),
        'tab_iop_all': [0] * (len(top_k) * len(thres) + 3),
        'tab_ans_all': [0] * (len(thres) + 5),
    }


def update_metrics_for_sample(metrics, sample, top_k, thres, cgbench_mode, conf_thr):
    """
    Update metrics dict for a single sample.

    metrics: dict from init_counters()
    sample: dict with keys
        - 'span'           : ground-truth [start, end]
        - 'pred_span'      : list of candidate spans [[s,e],...]
        - 'conf'           : list of confidences
        - 'pred_ori'       : (opt) original pred spans for comparison
        - 'conf_ori'       : (opt) original confidences
        - 'answer'         : predicted answer string
        - 'options'        : list of options
        - 'ans'            : ground-truth answer
        - 'task'           : task or list of tasks
        - 'grounder_success': bool
    top_k: list of ints, e.g. [1,3,5]
    thres: list of floats, IoU/IoP thresholds
    cgbench_mode: bool
    conf_thr: float
    """
    tab_iou = metrics['tab_iou']
    tab_iop = metrics['tab_iop']
    tab_ans = metrics['tab_ans']
    tab_iou_all = metrics['tab_iou_all']
    tab_iop_all = metrics['tab_iop_all']
    tab_ans_all = metrics['tab_ans_all']
    # global raises/lowers
    # NOTE: these four are simple ints in metrics
    # metrics['iou_raise'], metrics['iou_lower'], etc.

    # prepare tasks list
    tasks = sample.get('task', 'unknown')
    if isinstance(tasks, str):
        tasks = [tasks]

    # initialize counters for unseen tasks
    for t in tasks:
        if t not in tab_iou:
            tab_iou[t] = [0] * (len(top_k) * len(thres) + 3)  # 3 header: ['Task', '#Samples', 'Failed']
            tab_iop[t] = [0] * (len(top_k) * len(thres) + 3)
            tab_ans[t] = [0] * (len(thres) + 5)

    # 1) Grounding IoU & IoP
    if 'pred_span' in sample and 'span_conf' in sample and 'span' in sample:
        # count total samples
        for t in tasks:
            tab_iou[t][0] += 1
            tab_iop[t][0] += 1
        tab_iou_all[0] += 1
        tab_iop_all[0] += 1

        # compute IoU tensor
        iou_vals = compute_iou(
            sample['pred_span'],
            sample['span'],
            sample['span_conf'],
            cgbench_mode,
            conf_thr
        )
        top0 = float(iou_vals[0].max().item())
        # sum IoU scores
        for t in tasks:
            tab_iou[t][-1] += top0
        tab_iou_all[-1] += top0

        # recall@K@T
        iou_hit = [False] * (len(thres) + 1)
        for i, k in enumerate(top_k):
            max_k = float(iou_vals[:k].max().item())
            for j, thresh in enumerate(thres):
                if max_k >= thresh:
                    idx = i * len(thres) + j + 2
                    for t in tasks:
                        tab_iou[t][idx] += 1
                    tab_iou_all[idx] += 1
                    if k == 1:
                        iou_hit[j + 1] = True
                        if thresh == 0.5:
                            iou_hit[0] = True

        # compare with original
        if sample.get('pred_ori') is not None:
            iou_ori = compute_iou(
                sample['pred_ori'],
                sample['span'],
                sample['conf_ori'],
                cgbench_mode,
                conf_thr
            )
            ori0 = float(iou_ori[0].item())
            if ori0 < top0:
                metrics['iou_raise'] += 1
            if ori0 > top0:
                metrics['iou_lower'] += 1

        # IoP
        iop_vals = temporal_iof(
            torch.Tensor(sample['pred_span']),
            torch.Tensor(sample['span'])
        )
        iop_vals = torch.where(torch.isfinite(iop_vals), iop_vals, torch.zeros_like(iop_vals))
        top0_iop = float(iop_vals[0].max().item())
        for t in tasks:
            tab_iop[t][-1] += top0_iop
        tab_iop_all[-1] += top0_iop

        iop_hit = False
        for i, k in enumerate(top_k):
            max_k_iop = float(iop_vals[:k].max().item())
            for j, thresh in enumerate(thres):
                if max_k_iop >= thresh:
                    idx = i * len(thres) + j + 2
                    for t in tasks:
                        tab_iop[t][idx] += 1
                    tab_iop_all[idx] += 1
                    if k == 1 and thresh == 0.5:
                        iop_hit = True

        if sample.get('pred_ori') is not None:
            iop_ori = temporal_iof(
                torch.Tensor(sample['pred_ori']),
                torch.Tensor(sample['span'])
            )
            iop_ori = torch.where(torch.isfinite(iop_ori), iop_ori, torch.zeros_like(iop_ori))
            ori0_iop = float(iop_ori[0].item())
            if ori0_iop < top0_iop:
                metrics['iop_raise'] += 1
            if ori0_iop > top0_iop:
                metrics['iop_lower'] += 1

        # handle grounding failures
        if not sample.get('grounder_success', True):
            for t in tasks:
                tab_iou[t][1] += 1
                tab_iop[t][1] += 1
            tab_iou_all[1] += 1
            tab_iop_all[1] += 1
    else:
        # count as failure if no pred
        for t in tasks:
            tab_iou[t][1] += 1
            tab_iop[t][1] += 1
        tab_iou_all[1] += 1
        tab_iop_all[1] += 1

    # 2) QA metrics
    if 'answer' in sample and 'ans' in sample:
        for t in tasks:
            tab_ans[t][0] += 1
        tab_ans_all[0] += 1

        correct = check_ans(sample['options'], sample['ans'], sample['pred_answer'])
        if correct:
            for t in tasks:
                tab_ans[t][2] += 1
            tab_ans_all[2] += 1
            # combined IoU/IoP hits
            if 'iou_hit' in locals() and iou_hit[0]:
                for t in tasks:
                    tab_ans[t][3] += 1
                tab_ans_all[3] += 1
            if 'iop_hit' in locals() and iop_hit:
                for t in tasks:
                    tab_ans[t][4] += 1
                tab_ans_all[4] += 1
            # per-threshold IoU hits
            if 'iou_hit' in locals():
                for idx in range(1, len(iou_hit)):
                    if iou_hit[idx]:
                        for t in tasks:
                            tab_ans[t][idx + 4] += 1
                        tab_ans_all[idx + 4] += 1
        elif correct is None:
            for t in tasks:
                tab_ans[t][1] += 1
            tab_ans_all[1] += 1
    # end of function


def print_summary(metrics, top_k, thres, cgbench_mode):
    tab_iou = metrics['tab_iou']
    tab_iop = metrics['tab_iop']
    tab_ans = metrics['tab_ans']
    tab_iou_all = metrics['tab_iou_all']
    tab_iop_all = metrics['tab_iop_all']
    tab_ans_all = metrics['tab_ans_all']

    tasks = sorted(set(tab_iou.keys()) | set(tab_iop.keys()) | set(tab_ans.keys()))

    # Grounding (IoU)
    nncore.log('\nGrounding (IoU):')
    headers = ['Task', '#Samples', 'Failed'] + [f'R{k}@{t}' for k in top_k for t in thres]
    if cgbench_mode:
        headers += ['mIoU', 'rec.@IoU']
    else:
        headers += ['mIoU']
    rows = []
    for task in tasks:
        if task not in tab_iou:
            continue
        row = [task, tab_iou[task][0], tab_iou[task][1]]
        # R@K@T
        for i, k in enumerate(top_k):
            for j, _ in enumerate(thres):
                row.append(f'{tab_iou[task][i * len(thres) + j + 2] / tab_iou[task][0] * 100:.2f}')
        # mIoU
        if cgbench_mode:
            avg = sum(tab_iou[task][i] / tab_iou[task][0]
                      for i in range(2, 2 + len(thres))) / len(thres) * 100
            row += [f'{tab_iou[task][-1] / tab_iou[task][0] * 100:.2f}', f'{avg:.2f}']
        else:
            row += [f'{tab_iou[task][-1] / tab_iou[task][0] * 100:.2f}']
        rows.append(row)
    # all
    row_all = ['all', tab_iou_all[0], tab_iou_all[1]]
    for i, k in enumerate(top_k):
        for j, _ in enumerate(thres):
            row_all.append(f'{tab_iou_all[i * len(thres) + j + 2] / tab_iou_all[0] * 100:.2f}')
    if cgbench_mode:
        avg_all = sum(tab_iou_all[i] / tab_iou_all[0]
                      for i in range(2, 2 + len(thres))) / len(thres) * 100
        row_all += [f'{tab_iou_all[-1] / tab_iou_all[0] * 100:.2f}', f'{avg_all:.2f}']
    else:
        row_all += [f'{tab_iou_all[-1] / tab_iou_all[0] * 100:.2f}']
    rows.append(row_all)

    nncore.log(tabulate(rows, headers=headers, tablefmt='pretty', stralign='left'))

    # Grounding (IoP) – only in non-CG-Bench
    if not cgbench_mode:
        nncore.log('\nGrounding (IoP):')
        headers = ['Task', '#Samples', 'Failed'] + [f'R{k}@{t}' for k in top_k for t in thres] + ['mIoP']
        rows = []
        for task in tasks:
            if task not in tab_iop:
                continue
            row = [task, tab_iop[task][0], tab_iop[task][1]]
            for i, k in enumerate(top_k):
                for j, _ in enumerate(thres):
                    row.append(f'{tab_iop[task][i * len(thres) + j + 2] / tab_iop[task][0] * 100:.2f}')
            row += [f'{tab_iop[task][-1] / tab_iop[task][0] * 100:.2f}']
            rows.append(row)
        row_all = ['all', tab_iop_all[0], tab_iop_all[1]]
        for i, k in enumerate(top_k):
            for j, _ in enumerate(thres):
                row_all.append(f'{tab_iop_all[i * len(thres) + j + 2] / tab_iop_all[0] * 100:.2f}')
        row_all += [f'{tab_iop_all[-1] / tab_iop_all[0] * 100:.2f}']
        rows.append(row_all)

        nncore.log(tabulate(rows, headers=headers, tablefmt='pretty', stralign='left'))

    rows.append(row_all)

    nncore.log(tabulate(rows, headers=headers, tablefmt='pretty', stralign='left'))


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--pred_path')
    parser.add_argument('--dataset')
    parser.add_argument('--out_name', default='metrics.log')
    parser.add_argument('--conf_thr', type=float, default=-1)
    args = parser.parse_args()
    return args


if __name__ == '__main__':
    args = parse_args()

    assert nncore.is_dir(args.pred_path)
    pred_paths = nncore.ls(args.pred_path, ext=['json', 'jsonl'], join_path=True)

    log_file = nncore.join(args.pred_path, args.out_name)
    nncore.set_default_logger(logger='eval', fmt=None, log_file=log_file)

    cgbench_mode = (args.dataset == 'cgbench')
    top_k = [1] if cgbench_mode else [1, 3, 5]
    thres = [0.1, 0.2, 0.3, 0.4, 0.5] if cgbench_mode else [0.3, 0.5, 0.7]

    paths = args.paths.split(',')
    levels = args.levels.split(',')

    all_metrics = {}
    all_metrics = init_counters(top_k, thres)

    for path in pred_paths:
        data = nncore.load(path)
        for sample in data:
            update_metrics_for_sample(
                all_metrics,
                sample,
                top_k, thres,
                cgbench_mode,
                args.conf_thr
            )

    nncore.log(f"\n=== Results ===")
    print_summary(all_metrics, top_k, thres, cgbench_mode)
