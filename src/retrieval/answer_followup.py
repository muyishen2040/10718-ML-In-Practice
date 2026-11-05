"""Bounded answer-focused follow-up, using cached evidence rather than a corpus scan.

Question generation and span extraction never receive verdicts or annotated
answers. The extractive model is externally pretrained and stays frozen.
"""
from __future__ import annotations

import csv
import json
import math
from collections import Counter
from pathlib import Path

from src.data.corpus import canonical_url
from src.data.evidence import index_claims
from src.evaluation.splits import parse_claim_date, selection_split
from src.retrieval.answer_aware import best_extractable_span, source_excerpt
from src.retrieval.evidence_quality import EvidencePolicy, clean_candidates
from src.retrieval.question_guided import parse_questions
from src.utils.checkpoints import append_journal, atomic_jsonl, bind_run, read_journal
from src.utils.io import read_jsonl, write_json
from src.utils.progress import progress
from src.utils.runs import sha256_file


QUESTION_PROMPT = (
    "You break down a factual claim into two short, independently answerable "
    "verification questions. Do not decide if the claim is true. Do not use "
    'outside knowledge. Return JSON only: {"questions": ["...", "..."]}.'
)


def resolved_revision(model, requested, saved):
    if requested or saved:
        return requested or saved
    from huggingface_hub import model_info
    return model_info(model).sha


def valid_questions(value):
    return (isinstance(value, list) and len(value) == 2
            and all(isinstance(q, str) and 3 <= len(q.strip()) <= 400 for q in value)
            and value[0].strip().casefold() != value[1].strip().casefold())


def cached_questions(path, claims_path, records):
    """Accept only the known claim-only generator's artifacts with matching inputs."""
    if path is None or not path.is_file():
        return {}
    config_path = path.parent / "question_guided_config.json"
    if not config_path.is_file():
        raise ValueError(f"Question cache has no provenance: {config_path}")
    config = json.loads(config_path.read_text(encoding="utf-8"))
    if (config.get("script") != "run_question_guided_retrieval"
            or config.get("claims_sha256") != sha256_file(claims_path)
            or config.get("question_prompt") != QUESTION_PROMPT):
        raise ValueError(f"Question cache provenance/claim hash differs: {path}")
    expected = index_claims(records)
    result = {}
    for row in read_jsonl(path):
        cid = row.get("claim_id")
        if cid not in expected or cid in result or row.get("claim") != expected[cid]["claim"]:
            raise ValueError(f"Question cache claim ID/text mismatch: {cid}")
        if valid_questions(row.get("questions")):
            result[cid] = {"claim_id": cid, "claim": expected[cid]["claim"], "questions": row["questions"],
                           "question_origin": "previous_claim_only_generator", "fallback": row["questions"][0].startswith("What evidence supports")}
    return result


class QuestionGenerator:
    def __init__(self, name, revision, device, max_new_tokens):
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer
        torch.manual_seed(718)
        self.torch, self.device, self.max_new_tokens = torch, device, max_new_tokens
        self.tokenizer = AutoTokenizer.from_pretrained(name, revision=revision)
        self.model = AutoModelForCausalLM.from_pretrained(name, revision=revision,
            torch_dtype=torch.float16 if device.startswith("cuda") else torch.float32).to(device).eval()

    def __call__(self, claim):
        messages = [{"role": "system", "content": QUESTION_PROMPT}, {"role": "user", "content": claim}]
        text = self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True, enable_thinking=False)
        batch = self.tokenizer(text, return_tensors="pt").to(self.device)
        with self.torch.inference_mode():
            output = self.model.generate(**batch, do_sample=False, max_new_tokens=self.max_new_tokens,
                                         pad_token_id=self.tokenizer.eos_token_id)
        response = self.tokenizer.decode(output[0, batch.input_ids.shape[1]:], skip_special_tokens=True)
        questions = parse_questions(response, claim)
        fallback = not valid_questions(questions) or questions[0].startswith("What evidence supports")
        if not valid_questions(questions):
            questions = [f"What evidence supports or contradicts this claim: {claim[:200]}?",
                         f"Which facts establish the names, dates or numbers in this claim: {claim[:200]}?"]
        return questions, fallback


def prepare_questions(args, *, generator_factory=QuestionGenerator):
    config_path = args.output / "questions.json"
    previous = json.loads(config_path.read_text()) if config_path.is_file() else {}
    revision = resolved_revision(args.model, args.revision, previous.get("revision"))
    paths = {s: args.data_root / "processed/averitec" / f"{s}.jsonl" for s in ("train", "dev")}
    caches = {s: getattr(args, s + "_question_cache", None) for s in paths}
    config = {"version": 1, "model": args.model, "revision": revision, "prompt": QUESTION_PROMPT,
              "claim_hashes": {s: sha256_file(p) for s, p in paths.items()}, "seed": 718,
              "cache_hashes": {s: {"rows": sha256_file(p), "config": sha256_file(p.parent / "question_guided_config.json")}
                               for s, p in caches.items() if p is not None and p.is_file()},
              "max_new_tokens": args.max_new_tokens, "implementation_sha256": sha256_file(Path(__file__))}
    bind_run(config_path, config)
    generator = None
    totals = {}
    for split, path in paths.items():
        records = read_jsonl(path)
        expected = index_claims(records)
        journal = args.output / f"{split}_questions.partial.jsonl"
        done = read_journal(journal, set(expected))
        reusable = cached_questions(caches[split], path, records)
        with progress(total=len(records), initial=len(done), description=f"Claim-only questions / {split}", unit="claim") as bar:
            for record in records:
                cid = record["claim_id"]
                if cid in done:
                    continue
                if cid in reusable:
                    row = reusable[cid]
                else:
                    if generator is None:
                        print(f"Loading question generator {args.model}@{revision}", flush=True)
                        generator = generator_factory(args.model, revision, args.device, args.max_new_tokens)
                    questions, fallback = generator(record["claim"])
                    row = {"claim_id": cid, "claim": record["claim"], "questions": questions,
                           "question_origin": "new_claim_only_generator", "fallback": fallback}
                append_journal(journal, [row])
                done[cid] = row
                bar.update(1)
        ordered = [done[r["claim_id"]] for r in records]
        atomic_jsonl(ordered, args.output / f"{split}_questions.jsonl")
        totals[split] = {"claim_count": len(records), "fallback_count": sum(r["fallback"] for r in ordered),
                         "origins": dict(Counter(r["question_origin"] for r in ordered))}
    write_json({"config": config, "totals": totals, "finalized": True}, args.output / "questions_report.json")
    print(json.dumps(totals, indent=2))


def candidate_union(raw, selected):
    """At most six passages, with original top-three passages first."""
    result = {}
    for item in [*raw[:3], *selected[:3]]:
        pid = str(item.get("passage_id") or "")
        if not pid:
            raise ValueError("Missing passage ID")
        if pid in result and result[pid].get("text") != item.get("text"):
            raise ValueError(f"Same passage ID has different source text: {pid}")
        result.setdefault(pid, dict(item))
    return list(result.values())


class ExtractiveReader:
    def __init__(self, name, revision, device, batch_size, max_length):
        import torch
        from transformers import AutoModelForQuestionAnswering, AutoTokenizer
        self.torch, self.device, self.batch_size, self.max_length = torch, device, batch_size, max_length
        self.tokenizer = AutoTokenizer.from_pretrained(name, revision=revision, use_fast=True)
        if not self.tokenizer.is_fast:
            raise ValueError("Extractive offsets require a fast tokenizer")
        self.model = AutoModelForQuestionAnswering.from_pretrained(name, revision=revision).to(device).eval()

    def __call__(self, questions, candidates):
        jobs = [(pi, qi, question[:400], str(item.get("text") or ""))
                for pi, item in enumerate(candidates) for qi, question in enumerate(questions)]
        answers = [[] for _ in candidates]
        for start in range(0, len(jobs), self.batch_size):
            batch = jobs[start:start + self.batch_size]
            encoded = self.tokenizer([j[2] for j in batch], [j[3] for j in batch], padding=True,
                truncation="only_second", max_length=self.max_length, return_offsets_mapping=True, return_tensors="pt")
            offsets = encoded.pop("offset_mapping").tolist()
            segments = [encoded.sequence_ids(i) for i in range(len(batch))]
            with self.torch.inference_mode(), self.torch.autocast(device_type="cuda", dtype=self.torch.float16, enabled=self.device.startswith("cuda")):
                output = self.model(**{k: v.to(self.device) for k, v in encoded.items()})
            starts, ends = output.start_logits.float().cpu().tolist(), output.end_logits.float().cpu().tolist()
            for i, (pi, qi, question, text) in enumerate(batch):
                if not all(math.isfinite(v) for v in starts[i] + ends[i]):
                    raise ValueError("Nonfinite QA logits; cached rows are preserved")
                ids = encoded["input_ids"][i].tolist()
                cls_index = ids.index(self.tokenizer.cls_token_id)
                answer, margin = best_extractable_span(text, offsets[i], segments[i], starts[i], ends[i],
                    cls_index=cls_index, max_answer_tokens=24)
                answers[pi].append({"question_index": qi, "question": question, "answer": answer, "margin": margin})
        return answers


def pack_answers(item, answers):
    """Keep two questions and their exact source spans; never generate factual text."""
    text = str(item.get("text") or "")
    lines = []
    for answer in answers:
        span, margin = answer["answer"], answer["margin"]
        if span and span not in text:
            raise ValueError("Extracted answer is not a substring of its source")
        if not math.isfinite(margin):
            raise ValueError("Nonfinite answer-vs-null margin")
        confident = bool(span) and margin > 0
        lines += [f"Verification question: {answer['question']}",
                  f"Candidate source span: {span}" if confident else "No confident extractive answer for this question.",
                  "Verbatim source context: " + source_excerpt(text, span if confident else "", max_chars=220)]
    return {**item, "source_text": text, "text": "\n".join(lines), "extracted_answers": answers,
            "teacher_text_matches": False}


def select_answer_evidence(claim, date, candidates, answers):
    """Fixed coverage heuristic; no qrels, labels, learned weights or dev tuning."""
    packed = [pack_answers(p, a) for p, a in zip(candidates, answers, strict=True)]
    eligible, _ = clean_candidates(claim, date, candidates, EvidencePolicy(max_candidates=6, candidate_depth=6))
    eligible_ids = {p["passage_id"] for p in eligible}
    remaining = [p for p in packed if p["passage_id"] in eligible_ids]
    chosen, covered, sources = [], set(), set()
    for _ in range(min(3, len(remaining))):
        def score(p):
            answer_scores = [max(0.0, min(1.0, a["margin"] / 12)) if a["answer"] else 0.0 for a in p["extracted_answers"]]
            fresh = max((s for a, s in zip(p["extracted_answers"], answer_scores) if a["question_index"] not in covered), default=0.0)
            nli = p.get("nli_probabilities", {})
            stance = max(nli.get("entailment", 0), nli.get("contradiction", 0))
            source = canonical_url(p.get("url")) or p["passage_id"]
            prior = 1 / math.log2(max(1, int(p.get("original_rank", 100))) + 2)
            return max(answer_scores, default=0) + .35 * fresh + .15 * stance + .1 * prior + .1 * (source not in sources)
        best = sorted(remaining, key=lambda p: (-score(p), str(p["passage_id"])))[0]
        chosen.append({**best, "answer_selection_score": score(best)})
        sources.add(canonical_url(best.get("url")) or best["passage_id"])
        covered.update(a["question_index"] for a in best["extracted_answers"] if a["answer"] and a["margin"] > 0)
        remaining.remove(best)
    return chosen


def prepare_answers(args, *, reader_factory=ExtractiveReader):
    parent_path = args.parent_prepared / "preparation.json"
    parent = json.loads(parent_path.read_text(encoding="utf-8"))
    parent_report = json.loads((args.parent_prepared / "preparation_report.json").read_text(encoding="utf-8"))
    if parent.get("limit") or not parent_report.get("finalized"):
        raise ValueError("Reuse a fully finalized prior evidence preparation")
    config_path = args.output / "preparation.json"
    previous = json.loads(config_path.read_text()) if config_path.is_file() else {}
    revision = resolved_revision(args.model, args.revision, previous.get("qa_revision"))
    paths = {f"{s}_{kind}": path for s in ("train", "dev") for kind, path in (
        ("claims", args.data_root / "processed/averitec" / f"{s}.jsonl"),
        ("raw", args.parent_prepared / "raw" / f"{s}_rankings.jsonl"),
        ("clean_nli", args.parent_prepared / "clean_nli" / f"{s}_rankings.jsonl"),
        ("questions", args.questions / f"{s}_questions.jsonl"))}
    for s in ("train", "dev"):
        if parent["input_hashes"][s + "_claims"] != sha256_file(paths[s + "_claims"]):
            raise ValueError("Official claim input differs from cached NLI preparation")
    config = {"version": 2, "nli_model": parent["nli_model"], "revision": parent["revision"],
              "max_length": parent["max_length"], "claim_budget": parent["claim_budget"],
              "nli_label_order": parent.get("nli_label_order"), "limit": args.limit,
              "qa_model": args.model, "qa_revision": revision, "qa_max_length": args.max_length,
              "batch_size": args.batch_size, "parent_sha256": sha256_file(parent_path),
              "input_hashes": {k: sha256_file(p) for k, p in paths.items()},
              "question_config_sha256": sha256_file(args.questions / "questions.json"),
              "distillation_incompatible_modes": ["qa_same3", "qa_selected3"],
              "implementation_sha256": sha256_file(Path(__file__))}
    bind_run(config_path, config)
    reader, totals = None, {}
    for split in ("train", "dev"):
        all_records = read_jsonl(paths[split + "_claims"])
        all_ids = set(index_claims(all_records))
        records = all_records[:args.limit] if args.limit else all_records
        raw = index_claims(read_jsonl(paths[split + "_raw"]), expected=all_ids)
        selected = index_claims(read_jsonl(paths[split + "_clean_nli"]), expected=all_ids)
        questions = index_claims(read_jsonl(paths[split + "_questions"]), expected=all_ids)
        journal = args.output / f"{split}_answers.partial.jsonl"
        done = read_journal(journal, {r["claim_id"] for r in records})
        with progress(total=len(records), initial=len(done), description=f"Extract exact-source answers / {split}", unit="claim") as bar:
            for record in records:
                cid = record["claim_id"]
                if cid in done:
                    continue
                for source in (raw[cid], selected[cid], questions[cid]):
                    if source.get("claim") != record["claim"]:
                        raise ValueError(f"Claim text mismatch: {cid}")
                qs = questions[cid]["questions"]
                if not valid_questions(qs):
                    raise ValueError(f"Invalid question cache: {cid}")
                candidates = candidate_union(raw[cid]["retrieved"], selected[cid]["retrieved"])
                if candidates:
                    if reader is None:
                        print(f"Loading frozen extractive model {args.model}@{revision}", flush=True)
                        reader = reader_factory(args.model, revision, args.device, args.batch_size, args.max_length)
                    answers = reader(qs, candidates)
                    if len(answers) != len(candidates) or any(len(a) != 2 for a in answers):
                        raise ValueError("Need an answer result for both questions for every candidate")
                else:
                    answers = []
                packed = {p["passage_id"]: pack_answers(p, a) for p, a in zip(candidates, answers, strict=True)}
                date = parse_claim_date((record.get("metadata") or {}).get("claim_date"))
                row = {"claim_id": cid, "claim": record["claim"], "questions": qs, "raw": raw[cid]["retrieved"],
                       "qa_same3": [packed[p["passage_id"]] for p in raw[cid]["retrieved"][:3]],
                       "qa_selected3": select_answer_evidence(record["claim"], date, candidates, answers),
                       "candidate_count": len(candidates), "answered_pairs": sum(a["margin"] > 0 and bool(a["answer"]) for aa in answers for a in aa)}
                append_journal(journal, [row])
                done[cid] = row
                bar.update(1)
                bar.set_postfix(candidates=len(candidates), answered=row["answered_pairs"])
        ordered = [done[r["claim_id"]] for r in records]
        for mode in ("raw", "qa_same3", "qa_selected3"):
            atomic_jsonl([{"claim_id": r["claim_id"], "claim": r["claim"], "retrieved": r[mode]} for r in ordered],
                         args.output / mode / f"{split}_rankings.jsonl")
        totals[split] = {"claim_count": len(records), "candidate_count": sum(r["candidate_count"] for r in ordered),
                         "answered_pairs": sum(r["answered_pairs"] for r in ordered),
                         "empty_qa_selected3": sum(not r["qa_selected3"] for r in ordered)}
        if split == "train" and not args.limit:
            fit, _ = selection_split(records, .1, 718, "grouped")
            fit_ids = {records[int(i)]["claim_id"] for i in fit}
            audit = [{"claim_id": r["claim_id"], "claim": r["claim"], "passage_id": p["passage_id"], "url": p.get("url"),
                      "question": a["question"], "extracted_answer": a["answer"], "answer_margin": a["margin"],
                      "source_text": p["source_text"], "human_decisive": "", "human_note": ""}
                     for r in ordered if r["claim_id"] in fit_ids for p in r["qa_same3"] for a in p["extracted_answers"]]
            import random
            random.Random(718).shuffle(audit)
            with (args.output / "train_fit_answer_audit.csv").open("w", encoding="utf-8-sig", newline="") as handle:
                writer = csv.DictWriter(handle, fieldnames=list(audit[0]) if audit else ["claim_id"])
                writer.writeheader()
                writer.writerows(audit[:80])
    write_json({"config": config, "totals": totals, "finalized": True,
                "supervision": "External frozen QA; claim-only questions. Only the later verdict reader is trained on AVeriTeC labels.",
                "caveat": "Answer-vs-null margin is uncalibrated answerability, not truth or decisive relevance."}, args.output / "preparation_report.json")
    print(json.dumps(totals, indent=2))
