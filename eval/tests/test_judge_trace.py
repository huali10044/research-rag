"""
End-to-end scoring against the pinned RAGAS version with a stand-in judge.

No API key or network is used. The stand-in judge answers each RAGAS prompt
with valid JSON, except where a trigger word tells it to fail the way a real
judge does: a quota error, or text the parser cannot read. The assertions pin
down that each failure is recorded with its own reason instead of a bare NaN.
"""
import json

import pytest

import run_eval  # must precede ragas: it applies the Python 3.14 nest_asyncio fix

pytest.importorskip("ragas")

from langchain_core.embeddings import Embeddings  # noqa: E402
from langchain_core.language_models.chat_models import BaseChatModel  # noqa: E402
from langchain_core.messages import AIMessage  # noqa: E402
from langchain_core.outputs import ChatGeneration, ChatResult  # noqa: E402

METRICS = ["faithfulness", "answer_relevancy",
           "llm_context_precision_with_reference", "context_recall"]


class StandInJudge(BaseChatModel):
    @property
    def _llm_type(self) -> str:
        return "stand-in"

    def _generate(self, messages, stop=None, run_manager=None, **kwargs) -> ChatResult:
        prompt = "\n".join(str(m.content) for m in messages)
        if "QUOTATRIGGER" in prompt:
            raise RuntimeError("429 Resource exhausted: quota exceeded")
        if "GARBAGETRIGGER" in prompt:
            out = "I would rather not answer in JSON."
        elif "StatementGeneratorOutput" in prompt:
            out = json.dumps({"statements": ["s1", "s2"]})
        elif "NLIStatementOutput" in prompt:
            out = json.dumps({"statements": [
                {"statement": "s1", "reason": "r", "verdict": 1},
                {"statement": "s2", "reason": "r", "verdict": 0},
            ]})
        elif "ContextRecallClassifications" in prompt:
            out = json.dumps({"classifications": [
                {"statement": "a", "reason": "r", "attributed": 1},
            ]})
        elif "ResponseRelevanceOutput" in prompt:
            out = json.dumps({"question": "what?", "noncommittal": 0})
        else:
            out = json.dumps({"reason": "r", "verdict": 1})
        return ChatResult(generations=[ChatGeneration(message=AIMessage(content=out))])


class StandInEmbeddings(Embeddings):
    def embed_documents(self, texts):
        return [[1.0, 0.0, 1.0] for _ in texts]

    def embed_query(self, text):
        return [1.0, 0.0, 1.0]


def _record(qid, query, answer, contexts=("chunk a", "chunk b"), error=None,
            expected_type="content"):
    return {
        "id": qid, "query": query, "expected_type": expected_type,
        "actual_type": expected_type, "type_match": True,
        "reference": f"reference for {qid}", "must_contain": [],
        "answer": answer, "contexts": list(contexts),
        "similarities": [0.9] * len(contexts), "n_chunks": len(contexts),
        "latency_s": 0.1, "error": error,
    }


RECORDS = [
    _record("ok-1", "a plain question", "a plain answer"),
    _record("quota-1", "QUOTATRIGGER question", "QUOTATRIGGER answer"),
    _record("garbage-1", "GARBAGETRIGGER question", "a plain answer"),
    _record("generr-1", "another question", "", contexts=(), error="503 from generator"),
    _record("noctx-1", "a third question", "an answer", contexts=()),
]


@pytest.fixture
def stand_in(monkeypatch, tmp_path):
    from ragas.embeddings import LangchainEmbeddingsWrapper
    from ragas.llms import LangchainLLMWrapper
    import ragas.run_config as run_config_module

    real_run_config = run_config_module.RunConfig

    def fast_run_config(**kwargs):
        kwargs.update(max_retries=1, max_wait=1, timeout=30)
        return real_run_config(**kwargs)

    monkeypatch.setattr(run_config_module, "RunConfig", fast_run_config)
    monkeypatch.setattr(run_eval, "build_judge_llm",
                        lambda *a, **k: LangchainLLMWrapper(StandInJudge()))
    monkeypatch.setattr(run_eval, "build_embeddings",
                        lambda: LangchainEmbeddingsWrapper(StandInEmbeddings()))
    monkeypatch.setattr(run_eval, "RESULTS_DIR", tmp_path)
    monkeypatch.setattr(run_eval.rag, "EMBED_MODEL", "stand-in-embedder", raising=False)
    return tmp_path


@pytest.fixture
def track(stand_in):
    return run_eval.score_rag_track(RECORDS, "gemini", "stand-in-judge", max_workers=1)


def _row(track, qid):
    return next(r for r in track["rows"] if r["id"] == qid)


def test_scored_row_is_ok_on_every_metric(track):
    row = _row(track, "ok-1")
    assert all(row["status"][m] == "ok" for m in METRICS)
    assert row["faithfulness"] == 0.5
    assert "errors" not in row


def test_quota_failure_is_recorded_as_rate_limit(track):
    row = _row(track, "quota-1")
    assert all(row[m] is None for m in METRICS)
    assert all(row["status"][m] == "rate_limit" for m in METRICS)
    assert "429" in row["errors"]["context_recall"]


def test_unreadable_judge_output_is_recorded_as_parse_failure(track):
    row = _row(track, "garbage-1")
    # Answer relevancy only sees the answer, which carries no trigger word.
    assert row["status"]["answer_relevancy"] == "ok"
    for m in ("faithfulness", "llm_context_precision_with_reference", "context_recall"):
        assert row[m] is None
        assert row["status"][m] == "parse_failure"


def test_skipped_questions_carry_a_reason_and_count_against_coverage(track):
    assert track["skipped_reasons"] == {
        "generr-1": "generation_error",
        "noctx-1": "no_context",
    }
    cov = track["coverage"]["context_recall"]
    assert (cov["scored"], cov["total"]) == (1, 5)
    assert cov["missing"] == {
        "quota-1": "rate_limit",
        "garbage-1": "parse_failure",
        "generr-1": "generation_error",
        "noctx-1": "no_context",
    }
    assert track["complete"] is False


def test_every_judge_call_is_attributed_and_keeps_its_raw_output(track):
    calls = track["_judge_calls"]
    assert calls
    assert all(c["question_id"] in {"ok-1", "quota-1", "garbage-1"} for c in calls)
    assert all(c["metric"] in METRICS for c in calls)
    assert all(len(c["prompt_sha256"]) == 64 for c in calls)
    assert all("prompt_text" not in c for c in calls)

    raw = [c["output"] for c in calls
           if c["question_id"] == "garbage-1" and c["metric"] == "context_recall"]
    assert "I would rather not answer in JSON." in raw

    quota = [c for c in calls if c["question_id"] == "quota-1"]
    assert quota and all(c["output"] is None and "429" in c["error"] for c in quota)


def test_prompts_are_stored_only_on_request(stand_in):
    track = run_eval.score_rag_track(RECORDS[:1], "gemini", "stand-in-judge",
                                     max_workers=1, keep_prompts=True)
    import hashlib

    calls = track["_judge_calls"]
    assert calls
    for c in calls:
        assert len(c["prompt_text"]) == c["prompt_chars"] > 0
        assert hashlib.sha256(c["prompt_text"].encode()).hexdigest() == c["prompt_sha256"]
    assert any("a plain question" in c["prompt_text"] for c in calls)


def test_saved_run_can_be_replayed_without_regenerating(stand_in, monkeypatch):
    config = {"label": "top_k=8", "mode": "generate", "backend": "groq",
              "model": "gen-model", "top_k": 8,
              "judge_backend": "gemini", "judge_model": "stand-in-judge"}
    run = run_eval.score_records(RECORDS, config, track="rag", max_workers=1)

    assert run["config"]["judge_model"] == "stand-in-judge"
    assert run["config"]["judge_temperature"] == 0.0
    assert run["provenance"]["ragas"]
    assert len(run["provenance"]["golden_set_sha256"]) == 64
    assert run["provenance"]["question_ids"] == [r["id"] for r in RECORDS]

    json_path, md_path = run_eval.save_results([run], stamp="test")
    saved = json.loads(json_path.read_text())
    assert "_judge_calls" not in saved[0]
    assert saved[0]["records"][0]["contexts"] == ["chunk a", "chunk b"]

    judge_log = (stand_in / "eval-test.judge.jsonl").read_text().splitlines()
    assert len(judge_log) == len(run["_judge_calls"])
    assert json.loads(judge_log[0])["run"] == "top_k=8"
    assert "(1/5)" in md_path.read_text()

    # Replay must never call the generator.
    def no_generation(*args, **kwargs):
        raise AssertionError("replay regenerated answers")

    monkeypatch.setattr(run_eval, "generate_predictions", no_generation)

    [source] = run_eval.load_replay_runs(json_path)
    replay = run_eval.replay_one(source, json_path, "top_k=8, judge replay",
                                 track="rag", judge_backend="gemini",
                                 judge_model="stand-in-judge", max_workers=1)
    assert replay["config"]["mode"] == "replay"
    assert replay["config"]["source_label"] == "top_k=8"
    assert replay["config"]["top_k"] == 8
    assert _row(replay["rag_track"], "ok-1")["faithfulness"] == 0.5
    assert replay["rag_track"]["coverage"] == run["rag_track"]["coverage"]


def test_failure_before_any_judge_call_is_a_harness_error(stand_in, monkeypatch):
    """
    If a metric dies before calling the judge, that is not a judge result. This
    is what a broken event loop looks like (Python 3.14 with nest_asyncio
    applied): every score is NaN and no judge call is ever made.
    """
    from ragas.metrics import Faithfulness

    async def broken(self, sample, callbacks):
        raise RuntimeError("Timeout should be used inside a task")

    monkeypatch.setattr(Faithfulness, "_single_turn_ascore", broken)
    track = run_eval.score_rag_track(RECORDS[:1], "gemini", "stand-in-judge", max_workers=1)

    row = _row(track, "ok-1")
    assert row["faithfulness"] is None
    assert row["status"]["faithfulness"] == "harness_error"
    assert "Timeout should be used inside a task" in row["errors"]["faithfulness"]
    assert row["status"]["context_recall"] == "ok"
