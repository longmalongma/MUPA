# 这里假设四个 Agent 的实现与签名已就绪
from typing import List, Dict, Any
from abc import ABC, abstractmethod
from .agents_opt import GQAAgent, GrounderAgent, VerifierAgent, AnswererAgent, ReflectionAgent


class BasePath(ABC):
    def __init__(self,
                 gqa_agent: "GQAAgent",
                 grounder_agent: "GrounderAgent",
                 verifier_agent: "VerifierAgent",
                 answerer_agent: "AnswererAgent",
                 reflection_agent: "ReflectionAgent"):
        self.gqa = gqa_agent
        self.ground = grounder_agent
        self.verify = verifier_agent
        self.answer = answerer_agent
        self.reflect = reflection_agent

    @abstractmethod
    def run(self,
            video_path: str,
            question: str,
            query: str,
            duration: float,
            options: List[str] = None) -> Dict[str, Any]:
        """返回统一格式：
        {
            'answer': str,
            'span'  : List[float],
            'details': Dict[str, Any]  # 任意中间日志
        }
        """


class Path1_Original(BasePath):
    """
       question+query → Grounder → span → Verifier → span → Answerer → answer
    """

    def run(self, video_path, question, query, duration, options=None):
        details = {}
        # Grounder Agent
        grounder_out = self.ground.run(video_path=video_path, question=question, answer=None, query=query,
                                       duration=duration)
        details['grounder'] = grounder_out
        span_candidates = grounder_out['pred']
        conf_candidates = grounder_out['conf']

        # Verifier Agent
        # verifier_out = self.verify.run(video_path=video_path, question=question, answer=None, query=query,
        #                                candidates=span_candidates, conf=confs, duration=duration)
        # details['verifier'] = verifier_out
        # pred_spans = verifier_out['pred']
        # confs = verifier_out['conf']

        # Answerer Agent
        best_span = span_candidates[0] if span_candidates is not None else [0, duration]
        answerer_out = self.answer.run(video_path=video_path, question=question, selected_span=best_span,
                                       duration=duration, options=options)
        details['answerer'] = answerer_out
        answer = answerer_out['answer']

        # Reflection Agent
        reflection_out = self.reflect.run(mode="verify", video_path=video_path, question=question, answer=answer,
                                          query=query, pred_spans=span_candidates, pred_confs=conf_candidates,
                                          duration=duration)

        verifier_spans = reflection_out["verifier_spans"]
        verifier_confs = reflection_out["verifier_confs"]
        reflection_spans = reflection_out["reflection_spans"]
        reflection_confs = reflection_out["reflection_confs"]

        return dict(task="GQA", answer=answer, details=details, grounder_spans=span_candidates,
                    grounder_confs=conf_candidates, verifier_spans=verifier_spans, verifier_confs=verifier_confs,
                    reflection_spans=reflection_spans, reflection_confs=reflection_confs)


class Path2_GQA(BasePath):
    """② 先做 GQA（同时拿 answer + span），
       GQA -> span, answer + question 形成新 query -> Verifier
    """

    def run(self, video_path, question, query, duration, options=None):
        details = {}

        # GQA
        gqa_out = self.gqa.run(video_path, question, query, duration, options)
        details['gqa'] = gqa_out
        answer = gqa_out['answer']
        span_candidates = gqa_out['pred']
        conf_candidates = gqa_out['conf']

        # Verifier Agent
        # verifier_out = self.verify.run(video_path=video_path, question=question, answer=answer, query=query,
        #                                candidates=span_candidates, conf=gqa_confs, duration=duration)
        # details['verifier'] = verifier_out
        # pred_spans = verifier_out['pred']
        # ver_confs = verifier_out['conf']

        # Reflection Agent
        reflection_out = self.reflect.run(mode="verify", video_path=video_path, question=question, answer=answer,
                                          query=query, pred_spans=span_candidates, pred_confs=conf_candidates,
                                          duration=duration)

        verifier_spans = reflection_out["verifier_spans"]
        verifier_confs = reflection_out["verifier_confs"]
        reflection_spans = reflection_out["reflection_spans"]
        reflection_confs = reflection_out["reflection_confs"]

        return dict(task="GQA", answer=answer, details=details, gqa_spans=span_candidates,
                    gqa_confs=conf_candidates, verifier_spans=verifier_spans, verifier_confs=verifier_confs,
                    reflection_spans=reflection_spans, reflection_confs=reflection_confs)


class Path3_Answerer_First(BasePath):
    """③ 先让 Answerer 猜答案 → 拿答案+question 生成 query → Grounder → Verifier"""

    def run(self, video_path, question, query, duration, options=None):
        details = {}

        # Answerer Agent
        full_span = [0, duration]
        answerer_out = self.answer.run(video_path=video_path, question=question, selected_span=full_span,
                                       duration=duration, options=options)
        details['answerer'] = answerer_out
        answer = answerer_out['answer']

        # Grounder Agent
        grounder_out = self.ground.run(video_path=video_path, question=question, answer=answer, query=query,
                                       duration=duration)
        details['grounder'] = grounder_out
        span_candidates, conf_candidates = grounder_out['pred'], grounder_out['conf']

        # Verifier Agent
        # verifier_out = self.verify.run(video_path=video_path, question=question, answer=answer, query=query,
        #                                candidates=span_candidates, conf=confs, duration=duration)
        # details['verifier'] = verifier_out
        # pred_spans = verifier_out['pred']
        # confs = verifier_out['conf']

        # Reflection Agent
        reflection_out = self.reflect.run(mode="verify", video_path=video_path, question=question, answer=answer,
                                          query=query, pred_spans=span_candidates, pred_confs=conf_candidates,
                                          duration=duration)

        verifier_spans = reflection_out["verifier_spans"]
        verifier_confs = reflection_out["verifier_confs"]
        reflection_spans = reflection_out["reflection_spans"]
        reflection_confs = reflection_out["reflection_confs"]

        return dict(task="GQA", answer=answer, details=details, grounder_spans=span_candidates,
                    grounder_confs=conf_candidates, verifier_spans=verifier_spans, verifier_confs=verifier_confs,
                    reflection_spans=reflection_spans, reflection_confs=reflection_confs)


class Path4_MR(BasePath):
    """
    Specifically for Moment Retrieve task
    """

    def run(self, video_path, query, duration, question=None, options=None):
        details = {}
        # Grounder Agent
        grounder_out = self.ground.run(video_path=video_path, question=question, answer=None, query=query,
                                       duration=duration)
        details['grounder'] = grounder_out
        span_candidates, conf_candidates = grounder_out['pred'], grounder_out['conf']

        # Verifier Agent
        verifier_out = self.verify.run(video_path=video_path, question=question, answer=None, query=query,
                                       candidates=span_candidates, conf=conf_candidates, duration=duration)
        details['verifier'] = verifier_out
        verifier_spans = verifier_out['pred']
        verifier_confs = verifier_out['conf']

        return dict(task="MR", grounder_spans=span_candidates, grounder_confs=conf_candidates,
                    verifier_spans=verifier_spans, verifier_confs=verifier_confs, details=details)


class MultiPathInference_opt:
    def __init__(self, **kwargs):
        self.paths = {'path1': Path1_Original(**kwargs),
                      'path2': Path2_GQA(**kwargs),
                      'path3': Path3_Answerer_First(**kwargs),
                      'path4': Path4_MR(**kwargs)}

    def run(self, mode: str, *args, **kwargs) -> Dict[str, Any]:
        if mode not in self.paths:
            raise ValueError(f"Unknown path '{mode}'. choose from {list(self.paths)}")
        return self.paths[mode].run(*args, **kwargs)
