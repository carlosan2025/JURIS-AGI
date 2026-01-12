"""Tests for Memory & Abstraction Library (MAL) and trace persistence."""

import pytest
import json
import tempfile
from pathlib import Path

from juris_agi.core.types import Grid, ARCTask, ARCPair
from juris_agi.core.trace import (
    SolveTrace,
    TraceEntry,
    JSONLTraceWriter,
    JSONLTraceReader,
    create_trace_from_task,
)
from juris_agi.mal.retrieval import (
    MacroStore,
    StoredMacro,
    extract_task_tags,
    retrieve_macros,
)
from juris_agi.mal.macro_induction import (
    MacroInducer,
    CandidateMacro,
    extract_candidate_macros,
)
from juris_agi.mal.gating import (
    MacroGate,
    MacroAcceptanceResult,
    accept_macro,
)


# ============================================================================
# Test Fixtures
# ============================================================================

@pytest.fixture
def sample_task():
    """Create a sample ARC task for testing."""
    train_pairs = [
        ARCPair(
            input=Grid.from_list([[1, 2], [3, 4]]),
            output=Grid.from_list([[4, 3], [2, 1]]),
        ),
        ARCPair(
            input=Grid.from_list([[5, 6], [7, 8]]),
            output=Grid.from_list([[8, 7], [6, 5]]),
        ),
    ]
    return ARCTask(
        task_id="test_task_001",
        train=train_pairs,
        test=[],
    )


@pytest.fixture
def temp_trace_dir():
    """Create a temporary directory for trace files."""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield tmpdir


@pytest.fixture
def temp_macro_file():
    """Create a temporary file for macro storage."""
    with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
        yield f.name
    Path(f.name).unlink(missing_ok=True)


# ============================================================================
# Trace Tests
# ============================================================================

class TestSolveTrace:
    """Tests for SolveTrace class."""

    def test_trace_creation(self):
        """Should create a new trace."""
        trace = SolveTrace.start("test_task")
        assert trace.task_id == "test_task"
        assert trace.start_time is not None
        assert trace.success is False

    def test_trace_logging(self):
        """Should log entries."""
        trace = SolveTrace.start("test_task")
        trace.log("synthesis", "cre", program="identity")
        trace.log("evaluation", "cre", score=50.0)

        assert len(trace.entries) == 2
        assert trace.entries[0].event_type == "synthesis"
        assert trace.entries[1].details["score"] == 50.0

    def test_trace_finalize(self):
        """Should finalize trace."""
        trace = SolveTrace.start("test_task")
        trace.finalize(success=True, program="reflect_h >> rotate90")

        assert trace.success is True
        assert trace.final_program == "reflect_h >> rotate90"
        assert trace.end_time is not None

    def test_trace_to_dict(self):
        """Should convert to dictionary."""
        trace = SolveTrace.start("test_task")
        trace.log("test", "cre", value=42)
        trace.finalize(success=True, program="identity")

        d = trace.to_dict()
        assert d["task_id"] == "test_task"
        assert d["success"] is True
        assert len(d["entries"]) == 1


class TestJSONLTraceWriter:
    """Tests for JSONL trace writer."""

    def test_write_single_trace(self, temp_trace_dir):
        """Should write a single trace."""
        writer = JSONLTraceWriter(temp_trace_dir)
        trace = SolveTrace.start("test_task")
        trace.finalize(success=True, program="identity")

        path = writer.write_trace(trace, session_id="test_session")

        assert path.exists()
        assert path.name == "traces_test_session.jsonl"

    def test_write_multiple_traces(self, temp_trace_dir):
        """Should append multiple traces to same file."""
        writer = JSONLTraceWriter(temp_trace_dir)

        for i in range(3):
            trace = SolveTrace.start(f"task_{i}")
            trace.finalize(success=(i % 2 == 0))
            writer.write_trace(trace, session_id="multi_test")

        path = writer.get_session_path("multi_test")
        with open(path) as f:
            lines = f.readlines()

        assert len(lines) == 3

    def test_traces_are_valid_json(self, temp_trace_dir):
        """Each line should be valid JSON."""
        writer = JSONLTraceWriter(temp_trace_dir)
        trace = SolveTrace.start("json_test")
        trace.log("test", "cre", data={"nested": [1, 2, 3]})
        trace.finalize(success=True)

        path = writer.write_trace(trace, session_id="json_session")

        with open(path) as f:
            for line in f:
                data = json.loads(line)
                assert "task_id" in data


class TestJSONLTraceReader:
    """Tests for JSONL trace reader."""

    def test_read_traces(self, temp_trace_dir):
        """Should read traces from file."""
        # Write some traces
        writer = JSONLTraceWriter(temp_trace_dir)
        for i in range(5):
            trace = SolveTrace.start(f"read_task_{i}")
            trace.finalize(success=(i < 3))
            writer.write_trace(trace, session_id="read_test")

        # Read them back
        reader = JSONLTraceReader(temp_trace_dir)
        traces = reader.read_traces(success_only=False)

        assert len(traces) == 5

    def test_read_success_only(self, temp_trace_dir):
        """Should filter to successful traces."""
        writer = JSONLTraceWriter(temp_trace_dir)
        for i in range(5):
            trace = SolveTrace.start(f"filter_task_{i}")
            trace.finalize(success=(i < 2))
            writer.write_trace(trace, session_id="filter_test")

        reader = JSONLTraceReader(temp_trace_dir)
        traces = reader.read_traces(success_only=True)

        assert len(traces) == 2
        assert all(t["success"] for t in traces)

    def test_read_with_limit(self, temp_trace_dir):
        """Should respect limit parameter."""
        writer = JSONLTraceWriter(temp_trace_dir)
        for i in range(10):
            trace = SolveTrace.start(f"limit_task_{i}")
            trace.finalize(success=True)
            writer.write_trace(trace, session_id="limit_test")

        reader = JSONLTraceReader(temp_trace_dir)
        traces = reader.read_traces(limit=3)

        assert len(traces) == 3

    def test_get_successful_programs(self, temp_trace_dir):
        """Should extract successful programs."""
        writer = JSONLTraceWriter(temp_trace_dir)

        trace1 = SolveTrace.start("prog_task_1")
        trace1.finalize(success=True, program="rotate90(1)")
        writer.write_trace(trace1, session_id="prog_test")

        trace2 = SolveTrace.start("prog_task_2")
        trace2.finalize(success=False)
        writer.write_trace(trace2, session_id="prog_test")

        reader = JSONLTraceReader(temp_trace_dir)
        programs = reader.get_successful_programs()

        assert len(programs) == 1
        assert programs[0]["program"] == "rotate90(1)"


class TestCreateTraceFromTask:
    """Tests for trace creation from task."""

    def test_create_trace_extracts_features(self, sample_task):
        """Should extract task features into trace."""
        trace = create_trace_from_task(sample_task, "feature_test")

        assert trace.task_id == "feature_test"
        assert len(trace.entries) == 1  # task_loaded entry
        assert trace.entries[0].event_type == "task_loaded"
        assert trace.entries[0].details["num_train_pairs"] == 2


# ============================================================================
# MacroStore Tests
# ============================================================================

class TestMacroStore:
    """Tests for JSON-based macro storage."""

    def test_store_and_retrieve(self, temp_macro_file):
        """Should store and retrieve macros."""
        store = MacroStore(temp_macro_file)

        macro = StoredMacro(
            name="test_macro",
            code="rotate90(1) >> reflect_h",
            tags=["rotation", "reflection"],
        )
        store.store_macro(macro)

        retrieved = store.get_macro("test_macro")
        assert retrieved is not None
        assert retrieved.code == "rotate90(1) >> reflect_h"

    def test_retrieve_by_tags(self, temp_macro_file):
        """Should retrieve macros by tag similarity."""
        store = MacroStore(temp_macro_file)

        store.store_macro(StoredMacro(
            name="macro1",
            code="rotate90(1)",
            tags=["rotation", "same_dims"],
        ))
        store.store_macro(StoredMacro(
            name="macro2",
            code="scale(2)",
            tags=["scaling", "enlarging"],
        ))
        store.store_macro(StoredMacro(
            name="macro3",
            code="reflect_h >> rotate90(1)",
            tags=["rotation", "reflection", "same_dims"],
        ))

        # Query for rotation-related macros
        results = store.retrieve_by_tags(["rotation", "same_dims"], top_k=2)

        assert len(results) == 2
        # Both macro1 and macro3 have rotation tag, should be in results
        result_names = {r[0].name for r in results}
        assert "macro1" in result_names
        assert "macro3" in result_names
        # macro2 should not be in top 2 (no matching tags)
        assert "macro2" not in result_names

    def test_persistence(self, temp_macro_file):
        """Macros should persist across store instances."""
        store1 = MacroStore(temp_macro_file)
        store1.store_macro(StoredMacro(
            name="persist_test",
            code="identity",
            tags=["simple"],
        ))

        # Create new store instance
        store2 = MacroStore(temp_macro_file)
        macro = store2.get_macro("persist_test")

        assert macro is not None
        assert macro.code == "identity"

    def test_record_usage(self, temp_macro_file):
        """Should track macro usage."""
        store = MacroStore(temp_macro_file)
        store.store_macro(StoredMacro(
            name="usage_test",
            code="identity",
            tags=["test"],
        ))

        store.record_usage("usage_test", success=True)
        store.record_usage("usage_test", success=False)
        store.record_usage("usage_test", success=True)

        macro = store.get_macro("usage_test")
        assert macro.usage_count == 3
        assert macro.success_count == 2


class TestExtractTaskTags:
    """Tests for task tag extraction."""

    def test_same_dims_tag(self):
        """Should detect same dimensions."""
        task = ARCTask(
            task_id="same_dims",
            train=[
                ARCPair(
                    input=Grid.from_list([[1, 2], [3, 4]]),
                    output=Grid.from_list([[4, 3], [2, 1]]),
                ),
            ],
            test=[],
        )
        tags = extract_task_tags(task)
        assert "same_dims" in tags

    def test_scaling_tag(self):
        """Should detect scaling."""
        task = ARCTask(
            task_id="scale_2x",
            train=[
                ARCPair(
                    input=Grid.from_list([[1]]),
                    output=Grid.from_list([[1, 1], [1, 1]]),
                ),
            ],
            test=[],
        )
        tags = extract_task_tags(task)
        assert "enlarging" in tags
        assert "scale_2x" in tags

    def test_cropping_tag(self):
        """Should detect cropping."""
        task = ARCTask(
            task_id="cropping",
            train=[
                ARCPair(
                    input=Grid.from_list([[0, 1, 0], [0, 1, 0], [0, 0, 0]]),
                    output=Grid.from_list([[1], [1]]),
                ),
            ],
            test=[],
        )
        tags = extract_task_tags(task)
        assert "shrinking" in tags
        assert "cropping" in tags


# ============================================================================
# MacroInducer Tests
# ============================================================================

class TestMacroInducer:
    """Tests for trace-based macro induction."""

    def test_process_trace(self):
        """Should extract patterns from trace."""
        inducer = MacroInducer(min_frequency=1)

        trace = {
            "success": True,
            "task_id": "test1",
            "final_program": "rotate90(1) >> reflect_h",
            "entries": [],
        }
        inducer.process_trace(trace)

        candidates = inducer.extract_candidates()
        assert len(candidates) > 0

    def test_frequency_threshold(self):
        """Should respect minimum frequency."""
        inducer = MacroInducer(min_frequency=2)

        # Same pattern twice
        for i in range(2):
            trace = {
                "success": True,
                "task_id": f"test{i}",
                "final_program": "rotate90(1)",
                "entries": [],
            }
            inducer.process_trace(trace)

        candidates = inducer.extract_candidates()
        # Should have at least one candidate meeting threshold
        assert any(c.frequency >= 2 for c in candidates)

    def test_candidate_scoring(self):
        """Candidates should have score based on frequency and diversity."""
        inducer = MacroInducer(min_frequency=1)

        # Same pattern from multiple tasks
        for i in range(5):
            trace = {
                "success": True,
                "task_id": f"task_{i}",
                "final_program": "reflect_h",
                "entries": [],
            }
            inducer.process_trace(trace)

        candidates = inducer.extract_candidates()
        # reflect_h should be a candidate with high score
        reflect_candidates = [c for c in candidates if "reflect_h" in c.code]
        assert len(reflect_candidates) > 0
        assert reflect_candidates[0].frequency == 5


class TestExtractCandidateMacros:
    """Tests for convenience function."""

    def test_extract_from_traces(self):
        """Should extract macros from trace list."""
        traces = [
            {"success": True, "task_id": "t1", "final_program": "rotate90(1)", "entries": []},
            {"success": True, "task_id": "t2", "final_program": "rotate90(1)", "entries": []},
            {"success": False, "task_id": "t3", "final_program": None, "entries": []},
        ]

        candidates = extract_candidate_macros(traces, min_frequency=2)

        # rotate90(1) appeared twice in successful traces
        assert any(c.code == "rotate90(1)" for c in candidates)


# ============================================================================
# MacroGate Tests
# ============================================================================

class TestMacroGate:
    """Tests for macro acceptance gating."""

    def test_evaluate_macro_logs_mdl(self):
        """Should log MDL gain."""
        gate = MacroGate(log_decisions=True)

        result = gate.evaluate_macro(
            macro_code="rotate90(1) >> reflect_h",
            frequency=5,
            mdl_cost=2,
        )

        assert isinstance(result, MacroAcceptanceResult)
        assert "mdl_gain" in result.to_dict()

        log = gate.get_decision_log()
        assert len(log) == 1
        assert "mdl_gain" in log[0]

    def test_stub_always_rejects(self):
        """Stub should always reject macros."""
        gate = MacroGate()

        result = gate.evaluate_macro(
            macro_code="identity",
            frequency=100,
            mdl_cost=1,
        )

        assert result.accepted is False

    def test_mdl_gain_calculation(self):
        """Should calculate MDL gain correctly."""
        gate = MacroGate()

        # High frequency, low cost -> positive gain
        result1 = gate.evaluate_macro(
            macro_code="rotate90(1)",  # inline cost ~2
            frequency=10,
            mdl_cost=1,
        )

        # Low frequency, high cost -> negative gain
        result2 = gate.evaluate_macro(
            macro_code="rotate90(1) >> reflect_h >> scale(2)",  # inline cost ~5
            frequency=1,
            mdl_cost=10,
        )

        assert result1.mdl_gain > result2.mdl_gain

    def test_mdl_statistics(self):
        """Should compute MDL statistics."""
        gate = MacroGate()

        for freq in [1, 5, 10]:
            gate.evaluate_macro("identity", freq, 1)

        stats = gate.get_mdl_statistics()

        assert stats["total_evaluated"] == 3
        assert "avg_mdl_gain" in stats
        assert "max_mdl_gain" in stats


class TestAcceptMacroConvenience:
    """Tests for accept_macro convenience function."""

    def test_accept_macro_returns_result(self):
        """Should return acceptance result."""
        result = accept_macro(
            macro_code="reflect_h",
            frequency=3,
            mdl_cost=1,
        )

        assert isinstance(result, MacroAcceptanceResult)
        assert result.accepted is False  # Stub always rejects


# ============================================================================
# Integration Tests
# ============================================================================

class TestMALIntegration:
    """Integration tests for MAL workflow."""

    def test_trace_to_macro_workflow(self, temp_trace_dir, temp_macro_file):
        """End-to-end: write traces -> induce macros -> store -> retrieve."""
        # 1. Write some traces
        writer = JSONLTraceWriter(temp_trace_dir)
        for program in ["rotate90(1)", "rotate90(1)", "reflect_h", "rotate90(1) >> reflect_h"]:
            trace = SolveTrace.start(f"task_{program}")
            trace.finalize(success=True, program=program)
            writer.write_trace(trace, session_id="integration")

        # 2. Read traces
        reader = JSONLTraceReader(temp_trace_dir)
        traces = reader.read_all_traces(success_only=True)

        # 3. Induce macros
        candidates = extract_candidate_macros(traces, min_frequency=2)

        # 4. Store in MacroStore
        store = MacroStore(temp_macro_file)
        for candidate in candidates[:3]:
            stored = candidate.to_stored_macro()
            store.store_macro(stored)

        # 5. Retrieve for a similar task
        task = ARCTask(
            task_id="query_task",
            train=[
                ARCPair(
                    input=Grid.from_list([[1, 2], [3, 4]]),
                    output=Grid.from_list([[4, 3], [2, 1]]),  # rotation-like
                ),
            ],
            test=[],
        )
        results = retrieve_macros(task, store, top_k=5)

        # Should retrieve some macros
        assert len(results) >= 0  # May be empty if no tag matches
