from __future__ import annotations

from pathlib import Path

import pytest

from mcp_route_reliability.importers.jsonl import PredictionImportError, import_predictions
from mcp_route_reliability.models import RankedCandidate, RouterPrediction, ScoreMetadata

CORPUS_HASH = "sha256:corpus"
QUERY_HASH = "sha256:queries"
TOOLS = {"tool.a", "tool.b", "tool.c"}


def similarity_metadata() -> ScoreMetadata:
    return ScoreMetadata(
        score_type="similarity",
        higher_is_better=True,
        comparable_across_queries=False,
    )


def make_prediction(
    query_id: str,
    *,
    candidates: tuple[RankedCandidate, ...] | None = None,
    metadata: ScoreMetadata | None = None,
    corpus_hash: str = CORPUS_HASH,
    query_hash: str = QUERY_HASH,
    router_version: str = "1.0.0",
    model_version: str | None = "model:1",
    error: str | None = None,
) -> RouterPrediction:
    if candidates is None:
        candidates = (
            RankedCandidate(tool_id="tool.a", score=0.8, rank=1),
            RankedCandidate(tool_id="tool.b", score=0.4, rank=2),
        )
    return RouterPrediction(
        query_id=query_id,
        router_id="external",
        router_version=router_version,
        model_version=model_version,
        tool_corpus_hash=corpus_hash,
        query_set_hash=query_hash,
        score_metadata=metadata or similarity_metadata(),
        candidates=candidates,
        latency_ms=4.0,
        error=error,
    )


def write_predictions(path: Path, predictions: list[RouterPrediction]) -> None:
    path.write_text(
        "\n".join(prediction.model_dump_json() for prediction in predictions) + "\n",
        encoding="utf-8",
    )


def import_two(path: Path) -> list[RouterPrediction]:
    return import_predictions(
        path,
        expected_tool_corpus_hash=CORPUS_HASH,
        expected_query_set_hash=QUERY_HASH,
        expected_query_ids={"q1", "q2"},
        expected_tool_ids=TOOLS,
    )


def test_valid_predictions_are_returned_in_query_id_order(tmp_path: Path) -> None:
    path = tmp_path / "predictions.jsonl"
    write_predictions(path, [make_prediction("q2"), make_prediction("q1")])

    imported = import_two(path)

    assert [prediction.query_id for prediction in imported] == ["q1", "q2"]


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("corpus_hash", "sha256:wrong", "tool corpus hash"),
        ("query_hash", "sha256:wrong", "query set hash"),
    ],
)
def test_hash_mismatch_fails_before_results_are_returned(
    tmp_path: Path, field: str, value: str, message: str
) -> None:
    path = tmp_path / "predictions.jsonl"
    changes = {field: value}
    write_predictions(path, [make_prediction("q1", **changes), make_prediction("q2")])

    with pytest.raises(PredictionImportError, match=message):
        import_two(path)


@pytest.mark.parametrize(
    ("predictions", "message"),
    [
        ([make_prediction("q1")], "missing query IDs"),
        ([make_prediction("q1"), make_prediction("q2"), make_prediction("q3")], "extra query IDs"),
        ([make_prediction("q1"), make_prediction("q1")], "duplicate query_id"),
    ],
)
def test_query_id_set_must_match_exactly(
    tmp_path: Path, predictions: list[RouterPrediction], message: str
) -> None:
    path = tmp_path / "predictions.jsonl"
    write_predictions(path, predictions)
    with pytest.raises(PredictionImportError, match=message):
        import_two(path)


def test_unknown_candidate_tool_is_rejected(tmp_path: Path) -> None:
    path = tmp_path / "predictions.jsonl"
    unknown = (RankedCandidate(tool_id="tool.unknown", score=0.8, rank=1),)
    write_predictions(path, [make_prediction("q1", candidates=unknown), make_prediction("q2")])

    with pytest.raises(PredictionImportError, match="unknown Tool ID"):
        import_two(path)


@pytest.mark.parametrize("changed_field", ["router_version", "model_version"])
def test_router_contract_must_be_consistent(tmp_path: Path, changed_field: str) -> None:
    path = tmp_path / "predictions.jsonl"
    changes = {changed_field: "different"}
    write_predictions(path, [make_prediction("q1"), make_prediction("q2", **changes)])

    with pytest.raises(PredictionImportError, match="router/model contract"):
        import_two(path)


def test_score_metadata_must_be_consistent(tmp_path: Path) -> None:
    path = tmp_path / "predictions.jsonl"
    distance = ScoreMetadata(
        score_type="distance",
        higher_is_better=False,
        comparable_across_queries=False,
    )
    distance_candidates = (
        RankedCandidate(tool_id="tool.a", score=0.1, rank=1),
        RankedCandidate(tool_id="tool.b", score=0.2, rank=2),
    )
    write_predictions(
        path,
        [
            make_prediction("q1"),
            make_prediction("q2", metadata=distance, candidates=distance_candidates),
        ],
    )

    with pytest.raises(PredictionImportError, match="score metadata"):
        import_two(path)


def test_score_order_and_tie_break_are_validated(tmp_path: Path) -> None:
    path = tmp_path / "predictions.jsonl"
    wrong_order = (
        RankedCandidate(tool_id="tool.a", score=-2.0, rank=1),
        RankedCandidate(tool_id="tool.b", score=-1.0, rank=2),
    )
    write_predictions(
        path, [make_prediction("q1", candidates=wrong_order), make_prediction("q2")]
    )
    with pytest.raises(PredictionImportError, match="score ordering"):
        import_two(path)

    tied_wrong = (
        RankedCandidate(tool_id="tool.b", score=0.0, rank=1),
        RankedCandidate(tool_id="tool.a", score=0.0, rank=2),
    )
    write_predictions(
        path, [make_prediction("q1", candidates=tied_wrong), make_prediction("q2")]
    )
    with pytest.raises(PredictionImportError, match="tie ordering"):
        import_two(path)


def test_negative_singleton_and_rank_only_inputs_are_preserved(tmp_path: Path) -> None:
    path = tmp_path / "negative.jsonl"
    negative = (RankedCandidate(tool_id="tool.a", score=-1.5, rank=1),)
    write_predictions(path, [make_prediction("q1", candidates=negative)])
    imported = import_predictions(
        path,
        expected_tool_corpus_hash=CORPUS_HASH,
        expected_query_set_hash=QUERY_HASH,
        expected_query_ids={"q1"},
        expected_tool_ids=TOOLS,
    )
    assert imported[0].candidates[0].score == -1.5

    rank_path = tmp_path / "rank.jsonl"
    rank_metadata = ScoreMetadata(
        score_type="rank_only",
        higher_is_better=False,
        comparable_across_queries=False,
    )
    ranked = (
        RankedCandidate(tool_id="tool.a", score=None, rank=1),
        RankedCandidate(tool_id="tool.b", score=None, rank=2),
    )
    write_predictions(
        rank_path, [make_prediction("q1", metadata=rank_metadata, candidates=ranked)]
    )
    rank_imported = import_predictions(
        rank_path,
        expected_tool_corpus_hash=CORPUS_HASH,
        expected_query_set_hash=QUERY_HASH,
        expected_query_ids={"q1"},
        expected_tool_ids=TOOLS,
    )
    assert all(candidate.score is None for candidate in rank_imported[0].candidates)


def test_error_record_may_have_empty_candidates(tmp_path: Path) -> None:
    path = tmp_path / "error.jsonl"
    write_predictions(
        path,
        [make_prediction("q1", candidates=(), error="timeout")],
    )

    imported = import_predictions(
        path,
        expected_tool_corpus_hash=CORPUS_HASH,
        expected_query_set_hash=QUERY_HASH,
        expected_query_ids={"q1"},
        expected_tool_ids=TOOLS,
    )
    assert imported[0].error == "timeout"
    assert imported[0].candidates == ()


def test_corrupt_and_non_finite_json_names_file_and_line(tmp_path: Path) -> None:
    path = tmp_path / "broken.jsonl"
    path.write_text("not-json\n", encoding="utf-8")
    with pytest.raises(PredictionImportError, match=r"broken\.jsonl line 1"):
        import_predictions(
            path,
            expected_tool_corpus_hash=CORPUS_HASH,
            expected_query_set_hash=QUERY_HASH,
            expected_query_ids={"q1"},
            expected_tool_ids=TOOLS,
        )

    raw = make_prediction("q1").model_dump_json().replace('"score":0.8', '"score":NaN')
    path.write_text(raw + "\n", encoding="utf-8")
    with pytest.raises(PredictionImportError, match="finite"):
        import_predictions(
            path,
            expected_tool_corpus_hash=CORPUS_HASH,
            expected_query_set_hash=QUERY_HASH,
            expected_query_ids={"q1"},
            expected_tool_ids=TOOLS,
        )
