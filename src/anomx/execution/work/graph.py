"""Portable work graphs; the host decides how and where each step executes."""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from typing import Any, Literal


@dataclass(frozen=True)
class WorkStep:
    """One Python or inference step with an explicit runtime selection.

    ``task`` executes on the orchestrator, ``default`` uses the work's compute
    target, and other strings identify host-defined resources. The host must
    authorize those resources. Inference remains a distinct operation so a host
    can later reuse model sessions without changing the graph contract.
    """

    id: str
    kind: Literal["python", "inference"]
    code: str
    runtime: str = "task"
    dependencies: tuple[str, ...] = ()
    label: str = ""
    metadata: Mapping[str, Any] = field(default_factory=dict)


class WorkGraph:
    """A validated, deterministic DAG with explicit predecessor outputs.

    The executor receives a step and its inputs. ``previous`` is the sole direct
    predecessor's result (or a mapping for a join), ``results`` contains every
    completed step keyed by ID, and original inputs remain available. Execution
    is sequential in topological order; dispatch and isolation belong to the host.
    """

    def __init__(self, steps: list[WorkStep], *, output_node: str | None = None) -> None:
        if not steps or len(steps) > 100:
            raise ValueError("Work requires between 1 and 100 executable steps.")
        self.steps = tuple(steps)
        self.output_node = output_node
        by_id = {step.id: step for step in steps}
        if len(by_id) != len(steps) or any(not step.id for step in steps):
            raise ValueError("Step IDs must be nonempty and unique.")
        for step in steps:
            if step.kind not in {"python", "inference"}:
                raise ValueError("Steps must be Python or inference.")
            if not isinstance(step.code, str) or not step.code.strip() or len(step.code) > 200000:
                raise ValueError("Each step needs Python code of at most 200,000 characters.")
            compile(step.code, f"<work:{step.id}>", "exec")
            if not isinstance(step.runtime, str) or not step.runtime.strip():
                raise ValueError("Each step needs a runtime location.")
            if any(key not in by_id for key in step.dependencies):
                raise ValueError("Step dependencies must reference existing executable steps.")
        if output_node is not None and output_node not in by_id:
            raise ValueError("The output step must exist.")
        pending = dict(by_id)
        ordered = []
        while pending:
            ready = [
                step
                for step in pending.values()
                if all(key not in pending for key in step.dependencies)
            ]
            if not ready:
                raise ValueError("Work graphs cannot contain cycles.")
            for step in ready:
                ordered.append(step)
                del pending[step.id]
        self.ordered_steps = tuple(ordered)

    def execute(
        self,
        executor: Callable[[WorkStep, dict[str, Any]], Any],
        *,
        inputs: Mapping[str, Any] | None = None,
    ) -> object:
        """Execute every step once; propagate errors immediately to the host."""
        results: dict[str, Any] = {}
        for step in self.ordered_steps:
            predecessors = {key: results[key] for key in step.dependencies}
            previous = next(iter(predecessors.values())) if len(predecessors) == 1 else predecessors
            results[step.id] = executor(
                step, {**dict(inputs or {}), "previous": previous, "results": dict(results)}
            )
        return results[self.output_node] if self.output_node is not None else results

    @classmethod
    def from_dict(cls, graph: Mapping[str, Any]) -> WorkGraph:
        """Read the editor's start/code/inference/end representation.

        Legacy compute, log, notification and detection nodes remain readable.
        Start and end are structural nodes and cannot contain executable code.
        """
        nodes, edges = graph.get("nodes"), graph.get("edges", [])
        if not isinstance(nodes, list) or not 1 <= len(nodes) <= 100 or not isinstance(edges, list):
            raise ValueError("A graph requires 1–100 nodes and an edges list.")
        by_id: dict[str, dict[str, Any]] = {}
        for node in nodes:
            if (
                not isinstance(node, dict)
                or not isinstance(node.get("id"), str)
                or not node["id"]
                or node["id"] in by_id
            ):
                raise ValueError("Node IDs must be nonempty and unique.")
            by_id[node["id"]] = node
        incoming: dict[str, list[str]] = {key: [] for key in by_id}
        outgoing: dict[str, list[str]] = {key: [] for key in by_id}
        for edge in edges:
            if (
                not isinstance(edge, dict)
                or edge.get("source") not in by_id
                or edge.get("target") not in by_id
            ):
                raise ValueError("Connections must reference existing nodes.")
            source, target = edge["source"], edge["target"]
            if (
                source == target
                or by_id[source].get("type") == "end"
                or by_id[target].get("type") == "start"
            ):
                raise ValueError("Connections cannot loop or enter start or leave end.")
            if source not in incoming[target]:
                incoming[target].append(source)
                outgoing[source].append(target)
        starts = [key for key, node in by_id.items() if node.get("type") == "start"]
        ends = [key for key, node in by_id.items() if node.get("type") == "end"]
        if len(starts) != 1 or len(ends) != 1:
            raise ValueError("A work graph requires one start and one end.")
        for root, neighbors in ((starts[0], outgoing), (ends[0], incoming)):
            reached: set[str] = set()
            pending = [root]
            while pending:
                key = pending.pop()
                if key not in reached:
                    reached.add(key)
                    pending.extend(neighbors[key])
            if reached != set(by_id):
                raise ValueError("Every step must be connected from start to end.")
        steps = []
        for node in nodes:
            kind, config = node.get("type"), node.get("config", {})
            if not isinstance(config, dict):
                raise ValueError("Step configuration must be an object.")
            if kind in {"start", "end"}:
                continue
            code = config.get("code", "")
            if kind == "log":
                code = (
                    f"result = work.log({config.get('message', '')!r}, "
                    f"level={config.get('level', 'info')!r})"
                )
            elif kind == "notification":
                code = (
                    f"result = work.notify({config.get('title', 'Work notification')!r}, "
                    f"{config.get('message', '')!r})"
                )
            elif kind == "detection":
                code = (
                    f"result = work.detection({config.get('title', 'Detection')!r}, "
                    f"score={config.get('score')!r}, message={config.get('message', '')!r}, "
                    f"severity={config.get('severity', 'info')!r})"
                )
            elif kind not in {"python", "compute", "inference"}:
                raise ValueError(f"Unknown step type: {kind}.")
            steps.append(
                WorkStep(
                    id=node["id"],
                    kind="inference" if kind == "inference" else "python",
                    code=code,
                    runtime=config.get("runtime", "default" if kind == "compute" else "task"),
                    dependencies=tuple(key for key in incoming[node["id"]] if key not in starts),
                    label=str(node.get("label") or node["id"]),
                    metadata=dict(node.get("_meta") or {}),
                )
            )
        return cls(steps, output_node=graph.get("output_node"))
