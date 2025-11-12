"""
Class-based refactor for "LLM-Based Emotion Scoring (Debate→Converge)

This module is used to rate the emotion
- Integer scale 1–5 only
- Convergence if |S1 − S2| ≤ epsilon (default 1)
- Up to R debate rounds (default 3)
- Each model outputs S⁻, S, S⁺ and a brief reason no fabrication
- Final output per (id, emotion): {id, emotion, score_range:[L, U], converged: bool}
"""
from __future__ import annotations
from dotenv import load_dotenv
load_dotenv()
import argparse
import json
import os
import random
import re
import uuid
import time
import yaml
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

# Configuration

DEFAULT_R = 3
DEFAULT_EPSILON = 1
DEFAULT_TEMPERATURE = 0.2
DEFAULT_COOLDOWN = 0.6

@dataclass
class RunConfig:
    provider1: str = ""
    model1: str = ""
    provider2: str = ""
    model2: str = ""
    input: str = ""
    output: str = ""
    rubrics: str = ""
    temperature: float = DEFAULT_TEMPERATURE
    R: int = DEFAULT_R
    epsilon: int = DEFAULT_EPSILON
    cooldown: float = DEFAULT_COOLDOWN
    emotions: Optional[List[str]] = None
    rubric_path: Optional[str] = None

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "RunConfig":

        
        import dataclasses
        field_names = {f.name for f in dataclasses.fields(cls)}
        
        kwargs = {k: v for k, v in d.items() if k in field_names}
        
        return cls(**kwargs)

    @classmethod
    def from_yaml(cls, path: str) -> "RunConfig":
        if yaml is None:
            raise RuntimeError("PyYAML not installed. pip install pyyaml")
        with open(path, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f) or {}
        return cls.from_dict(data)

# LLM Client 
from src.components.llm_call import handler  

def make_client(provider: str, model: str, temperature: float) -> "LLMClient":
    return LLMClient(provider=provider, model=model, temperature=temperature)

class LLMClient:
    def __init__(self, provider: str, model: str, temperature: float):
        self.provider = provider
        self.model = model
        self.temperature = temperature

    def generate(self, prompt: str, seed: Optional[int] = 42) -> str:
        try:
            resp = handler.call_llm(
                provider=self.provider,
                prompt=prompt,
                model=self.model,
                temperature=self.temperature,
                stream=False,
                system_prompt=None,
                history=None,
            )
        except Exception as e:
            print(f"[Error calling {self.provider}/{self.model}] {e}")
            return ""

        if isinstance(resp, str):
            return resp
        return json.dumps(resp, ensure_ascii=False)


# Parsing utilities

ParsedBlock = Dict[str, Any]

class BlockParser:
    SCORE_KEYS = ("Sminus", "S", "Splus")

    @staticmethod
    def parse_block(raw: str) -> ParsedBlock:
        """Parse a block containing Emotion/S⁻/S/S⁺/Reason."""
        data: ParsedBlock = {"emotion": None, "Sminus": None, "S": None, "Splus": None, "Reason": ""}
        lines = [ln.strip() for ln in raw.strip().splitlines() if ln.strip()]
        for ln in lines:
            low = ln.lower()
            if low.startswith("emotion:"):
                data["emotion"] = ln.split(":", 1)[1].strip()
            elif low.startswith("s⁻:") or low.startswith("s-:"):
                m = re.search(r"(\d+)", ln)
                if m:
                    data["Sminus"] = int(m.group(1))
            elif low.startswith("s+:"):
                m = re.search(r"(\d+)", ln)
                if m:
                    data["Splus"] = int(m.group(1))
            elif low.startswith("s:"):
                m = re.search(r"(\d+)", ln)
                if m:
                    data["S"] = int(m.group(1))
            elif low.startswith("reason:"):
                data["Reason"] = ln.split(":", 1)[1].strip()
        # Guardrails: ensure integers in [1,5]
        for k in BlockParser.SCORE_KEYS:
            v = data.get(k)
            if v is None:
                continue
            try:
                iv = int(v)
            except Exception:
                iv = 3
            data[k] = max(1, min(5, iv))
        # Enforce S- ≤ S ≤ S+
        a, b, c = data.get("Sminus") or 3, data.get("S") or 3, data.get("Splus") or 3
        if a > b:
            a = b
        if c < b:
            c = b
        data["Sminus"], data["S"], data["Splus"] = a, b, c
        return data


# Prompts 
USER_SCORING_PROMPT = (
    "You are a third-person emotional appraisal analyst.\n"
    "Task: assess how strongly general readers would feel the target emotion after reading the article.\n"
    "Rules: Only use information from the input text. Do NOT invent or assume any facts not present in the text.\n"
    "Scale: integers 1–5 ONLY.\n\n"
    "INPUT\n"
    "Emotion label: {emotion}\n"
    "Emotion definition & rubric (1–5):\n{rubric}\n\n"
    "Article:\n{article}\n\n"
    "OUTPUT FORMAT (strict)\n"
    "Emotion: {emotion}\n"
    "S⁻: [integer 1–5]\n"
    "S: [integer 1–5]\n"
    "S⁺: [integer 1–5]\n"
    "Reason: [≤3 sentences; state what in the text triggers the emotion, why S is justified, "
    "and that weakening/strengthening would lower/raise to S⁻/S⁺; must satisfy S⁻ ≤ S ≤ S⁺]\n"
)

REVIEW_PROMPT = (
    "You are reviewing another model's rating for the SAME article and emotion.\n"
    "Rules: Use ONLY the article's content. Do NOT invent details.\n"
    "Other model's last rating:\n"
    "Emotion: {emotion}\n"
    "S⁻: {p_Sminus}\n"
    "S: {p_S}\n"
    "S⁺: {p_Splus}\n"
    "Reason: {p_reason}\n\n"
    "Article:\n{article}\n\n"
    "First, write one word: agree / uncertain / disagree, then ≤2 sentences why.\n"
    "Then provide your own revised rating in STRICT format:\n"
    "Emotion: {emotion}\n"
    "S⁻: [integer 1–5]\n"
    "S: [integer 1–5]\n"
    "S⁺: [integer 1–5]\n"
    "Reason: [≤3 sentences; must justify S; and indicate how weakening/strengthening maps to S⁻/S⁺; S⁻ ≤ S ≤ S⁺]\n"
)


# Debate/Convergence Engine

@dataclass
class DebateResult:
    id: str
    emotion: str
    score_range: List[int]  # [L,U]
    converged: bool


class DebateEngine:
    def __init__(self, cfg: RunConfig, client1: LLMClient, client2: LLMClient):
        self.cfg = cfg
        self.c1 = client1
        self.c2 = client2

    def _is_converged(self, s1: int, s2: int) -> bool:
        return abs(int(s1) - int(s2)) <= int(self.cfg.epsilon)

    def _score_once(self, client: LLMClient, article: str, emotion: str, rubric: str, seed: int) -> ParsedBlock:
        prompt = USER_SCORING_PROMPT.format(emotion=emotion, rubric=rubric, article=article)
        raw = client.generate(prompt, seed=seed)
        return BlockParser.parse_block(raw)

    def _revise(self, client: LLMClient, article: str, emotion: str, rubric: str, other: ParsedBlock, seed: int) -> ParsedBlock:
        other_str = REVIEW_PROMPT.format(
            emotion=emotion,
            p_Sminus=other.get("Sminus", 3),
            p_S=other.get("S", 3),
            p_Splus=other.get("Splus", 3),
            p_reason=other.get("Reason", ""),
            article=article,
        )
        raw = client.generate(other_str, seed=seed)
        return BlockParser.parse_block(raw)

    def run_pair(self, article: str, emotion: str, rubric: str, seed_base: int = 7) -> DebateResult:
        # Round 1
        out1 = self._score_once(self.c1, article, emotion, rubric, seed=seed_base)
        out2 = self._score_once(self.c2, article, emotion, rubric, seed=seed_base + 1)
        s1, s2 = out1["S"], out2["S"]
        if self._is_converged(s1, s2):
            return DebateResult(id="", emotion=emotion, score_range=[min(s1, s2), max(s1, s2)], converged=True)

        # Rounds 2..R
        for r in range(2, int(self.cfg.R) + 1):
            out1 = self._revise(self.c1, article, emotion, rubric, other=out2, seed=seed_base + 10 * r)
            out2 = self._revise(self.c2, article, emotion, rubric, other=out1, seed=seed_base + 10 * r + 1)
            s1, s2 = out1["S"], out2["S"]
            if self._is_converged(s1, s2):
                return DebateResult(id="", emotion=emotion, score_range=[min(s1, s2), max(s1, s2)], converged=True)

        return DebateResult(id="", emotion=emotion, score_range=[min(s1, s2), max(s1, s2)], converged=False)

# I/O utilities


def read_rows(in_path: Path) -> List[Dict[str, Any]]:
    if in_path.suffix == ".json":
        with open(in_path, "r", encoding="utf-8") as f:
            return json.load(f)
    elif in_path.suffix == ".jsonl":
        rows = []
        with open(in_path, "r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    rows.append(json.loads(line))
        return rows
    else:
        raise ValueError(f"Unsupported input format: {in_path.suffix}")

def write_jsonl(path: Path, records: Iterable[Dict[str, Any]]) -> None:
    with open(path, "w", encoding="utf-8") as f:
        for rec in records:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")

def load_rubric(rubrics_arg: str) -> Dict[str, str]:
    """rubrics_arg can be a JSON file, a JSON string, or a plain text file used for all emotions."""
    p = Path(rubrics_arg)
    if p.exists():
        text = p.read_text(encoding="utf-8")
        try:
            data = json.loads(text)
            if isinstance(data, dict):
                return {str(k): str(v) for k, v in data.items()}
        except Exception:
            pass
        # treat as a single rubric
        return {"*": text}
    # not a file → try JSON literal
    try:
        data = json.loads(rubrics_arg)
        if isinstance(data, dict):
            return {str(k): str(v) for k, v in data.items()}
    except Exception:
        pass
    # fall back to same rubric for all emotions
    return {"*": str(rubrics_arg)}


# Pipeline


class ScoringPipeline:
    def __init__(self, cfg: RunConfig, engine: DebateEngine):
        self.cfg = cfg
        self.engine = engine
        
        
    def _resolve_emotions(self, row: Dict[str, Any], override: Optional[List[str]]) -> List[str]:
        if override:
            return [str(x).strip() for x in override if str(x).strip()]
        if "emotions" in row and row["emotions"]:
            try:
                vals = row["emotions"]
                if isinstance(vals, str) and vals.strip().startswith("["):
                    return [str(x).strip() for x in json.loads(vals)]
                if isinstance(vals, (list, tuple)):
                    return [str(x).strip() for x in vals]
            except Exception:
                pass
            return self.cfg.emotions or []

    def run(self, rows: List[Dict[str, Any]], rubrics: Dict[str, str], override_emotions: Optional[List[str]] = None) -> List[Dict[str, Any]]:
        outputs: List[Dict[str, Any]] = []
        for i, row in enumerate(rows):
            rid = str(row.get("id", i))
            article = str(row.get("text", ""))
            emotions = self._resolve_emotions(row, override_emotions)
            for emo in emotions:
                rubric = rubrics.get(emo, rubrics.get("*", "Rate on the 1–5 integer scale."))
                res = self.engine.run_pair(article, emo, rubric, seed_base=101 + i)
                outputs.append({
                    "id": rid,
                    "emotion": emo,
                    "score_range": res.score_range,
                    "converged": res.converged,
                })
                time.sleep(float(self.cfg.cooldown))
        return outputs


# CLI

def build_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Emotion scoring (debate→converge) – class refactor")
    p.add_argument("--input", required=None, help="(Optional)Input CSV/JSONL with columns: id,text")
    p.add_argument("--output", required=None, help="(Optional)Output JSONL path")
    p.add_argument("--rubrics", required=None, help="(Optional)Rubrics file or JSON (dict) or text")
    p.add_argument("--emotions", default="", help="(Optional)Override emotions: JSON list or comma-list")

    p.add_argument("--provider1", default ="glm" )
    p.add_argument("--model1", default="glm-4-air")
    p.add_argument("--provider2", default="glm")
    p.add_argument("--model2", default="glm-4-plus-mini")

    p.add_argument("--temperature", type=float, default=DEFAULT_TEMPERATURE)
    p.add_argument("--R", type=int, default=DEFAULT_R)
    p.add_argument("--epsilon", type=int, default=DEFAULT_EPSILON)
    p.add_argument("--cooldown", type=float, default=DEFAULT_COOLDOWN)

    p.add_argument("--config_yaml", default="config/llm_labeler.yaml", help="Optional YAML to override all settings")
    return p


def main(argv: Optional[List[str]] = None) -> int:
    args = build_argparser().parse_args(argv)
    cfg = RunConfig.from_yaml(args.config_yaml) if args.config_yaml else RunConfig() 
    
    c1 = make_client(cfg.provider1, cfg.model1, cfg.temperature)
    c2 = make_client(cfg.provider2, cfg.model2, cfg.temperature)
    engine = DebateEngine(cfg, c1, c2)
    
    in_path = Path(cfg.input)
    out_path = Path(cfg.output)
    rows = read_rows(in_path) 
    
    # emotions override
    override_emotions: List[str] = []
    if args.emotions.strip():
        try:
            override_emotions = [str(x).strip() for x in json.loads(args.emotions)]
        except Exception:
            override_emotions = [x.strip() for x in args.emotions.split(",") if x.strip()]

    # rubrics
    rubrics_map = load_rubric(cfg.rubrics)

    # pipeline
    pipeline = ScoringPipeline(cfg, engine)
    outputs = pipeline.run(rows, rubrics_map, override_emotions or cfg.emotions)

    # write
    out_path.parent.mkdir(parents=True, exist_ok=True)
    write_jsonl(out_path, outputs)
    print(f"[OK] wrote {len(outputs)} records → {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
