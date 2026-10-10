import base64
import io

import numpy as np
import pandas as pd
import pytest

from anomx import WorkContext
from anomx.work import WorkGraph, WorkStep
from anomx.work.values import decode_work_value, encode_work_value


def graph(nodes, edges):
    return {
        "nodes": [{"id": "start", "type": "start"}, *nodes, {"id": "end", "type": "end"}],
        "edges": [{"source": a, "target": b} for a, b in edges],
    }


def test_graph_transfers_step_results_and_dispatches_compute():
    calls = []

    def compute(**arguments):
        calls.append(arguments)
        return arguments["inputs"]["previous"] + arguments["inputs"]["seed"]

    definition = graph(
        [
            {"id": "one", "type": "python", "config": {"code": 'result = inputs["seed"] * 2'}},
            {
                "id": "two",
                "type": "inference",
                "config": {"code": "result = 0", "runtime": "service:gpu"},
            },
        ],
        [("start", "one"), ("one", "two"), ("two", "end")],
    )
    work = WorkContext(callbacks={"compute": compute}, inputs={"seed": 3})
    assert work.run_graph(definition) == {"one": 6, "two": 9}
    assert calls[0]["target"] == "service:gpu"
    assert calls[0]["inputs"]["results"] == {"one": 6}


def test_compute_inherits_context_unless_explicitly_replaced():
    work = WorkContext(
        callbacks={"compute": lambda **args: args}, inputs={"dataset_reference": "data_dataset-1"}
    )
    assert work.compute("result = 1")["inputs"] == work.inputs
    assert work.compute("result = 1", inputs={})["inputs"] == {}


def test_graph_rejects_cycles_and_disconnected_nodes():
    with pytest.raises(ValueError, match="cycles"):
        WorkGraph(
            [
                WorkStep("a", "python", "result = 1", dependencies=("b",)),
                WorkStep("b", "python", "result = 2", dependencies=("a",)),
            ]
        )
    with pytest.raises(ValueError, match="connected"):
        WorkGraph.from_dict(
            graph(
                [{"id": "a", "type": "python", "config": {"code": "result = 1"}}],
                [("start", "end")],
            )
        )


def test_transport_preserves_arrays_tables_and_rejects_pickle():
    value = {
        "array": np.arange(12, dtype=np.float32).reshape(3, 4),
        "table": pd.DataFrame({"x": [1.0, 2.0]}),
    }
    decoded = decode_work_value(encode_work_value(value))
    np.testing.assert_array_equal(decoded["array"], value["array"])
    pd.testing.assert_frame_equal(decoded["table"], value["table"])
    with pytest.raises(ValueError, match="objects"):
        encode_work_value(np.array([object()], dtype=object))
    with pytest.raises(TypeError, match="Unsupported"):
        encode_work_value(object())


def test_transport_rejects_oversized_array_header_before_allocation():
    stream = io.BytesIO()
    np.lib.format.write_array_header_1_0(
        stream, {"descr": "<f8", "fortran_order": False, "shape": (10**12,)}
    )
    with pytest.raises(ValueError, match="size"):
        decode_work_value(
            {
                "__anomx_work_value__": "ndarray",
                "data": base64.b64encode(stream.getvalue()).decode(),
            }
        )
