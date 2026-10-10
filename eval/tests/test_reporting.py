"""
Coverage, same-question comparison and replay loading.

These run against the committed September sweep, which is the case that
motivated them: k=8 run 1 was scored on 9 of 18 questions, and its 0.833
recall was read as a peak against runs scored on all 18.
"""
import json

import pytest

import run_eval

SWEEP = run_eval.RESULTS_DIR / "eval-sweep-combined.json"


@pytest.fixture(scope="module")
def sweep():
    return json.loads(SWEEP.read_text())


def _run(sweep, label):
    return next(r for r in sweep if r["config"]["label"] == label)


def test_coverage_exposes_the_half_scored_run(sweep):
    cov = run_eval.metric_coverage(_run(sweep, "top_k=8 (run 1)")["rag_track"])
    assert cov["context_recall"]["scored"] == 9
    assert cov["context_recall"]["total"] == 18
    # The file predates reason tracking, and the harness says so.
    assert set(cov["context_recall"]["missing"].values()) == {"unrecorded"}
    assert not run_eval.is_complete(_run(sweep, "top_k=8 (run 1)")["rag_track"])
    assert run_eval.is_complete(_run(sweep, "top_k=8 (run 2)")["rag_track"])


def test_same_question_means_remove_the_peak(sweep):
    ids, means = run_eval.matched_question_means(sweep, "context_recall")
    assert len(ids) == 9
    by_label = dict(zip([r["config"]["label"] for r in sweep], means))
    assert by_label["top_k=3"] == pytest.approx(0.6667, abs=1e-4)
    assert by_label["top_k=5 (run 1)"] == pytest.approx(0.7778, abs=1e-4)
    assert by_label["top_k=8 (run 1)"] == pytest.approx(0.8333, abs=1e-4)
    assert by_label["top_k=8 (run 2)"] == pytest.approx(0.7222, abs=1e-4)
    assert by_label["top_k=8 (run 3)"] == pytest.approx(0.7778, abs=1e-4)
    assert by_label["top_k=12"] == pytest.approx(0.7222, abs=1e-4)


def test_report_marks_partial_means(sweep):
    md = run_eval.render_markdown(sweep)
    assert "0.833 (9/18)" in md
    assert "## Incomplete coverage" in md
    assert "## Same-question comparison" in md
    # A fully scored run keeps a bare number.
    prefix = "| top_k=8 (run 2) | 18 |"
    row = next(line for line in md.splitlines() if line.startswith(prefix))
    assert "(" not in row[len(prefix):]


def test_strict_withholds_partial_means(sweep):
    md = run_eval.render_markdown(sweep, strict=True)
    assert "withheld (9/18)" in md
    assert "0.833 (9/18)" not in md


def test_complete_sweep_gets_no_coverage_sections(sweep):
    full = [_run(sweep, "top_k=5 (run 1)"), _run(sweep, "top_k=8 (run 2)")]
    md = run_eval.render_markdown(full)
    assert "Incomplete coverage" not in md
    assert "Same-question comparison" not in md


@pytest.mark.parametrize(
    "type_name, message, expected",
    [
        ("RagasOutputParserException", "The output parser failed to parse the output", "parse_failure"),
        ("ResourceExhausted", "429 You exceeded your current quota", "rate_limit"),
        ("RuntimeError", "Too Many Requests", "rate_limit"),
        ("TimeoutError", "", "timeout"),
        ("ValueError", "something else", "judge_error"),
    ],
)
def test_classify_judge_error(type_name, message, expected):
    assert run_eval.classify_judge_error(type_name, message) == expected


def test_replay_refuses_files_saved_without_contexts():
    with pytest.raises(SystemExit) as exc:
        run_eval.load_replay_runs(SWEEP)
    assert "without the retrieved contexts" in str(exc.value)


def test_replay_label_must_exist():
    with pytest.raises(SystemExit):
        run_eval.load_replay_runs(SWEEP, label="top_k=99")
