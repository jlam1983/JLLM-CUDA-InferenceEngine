"""
JLLM Computation Graph
=====================
Defines the dependency graph of transformer layer computations.
"""

from collections import defaultdict, deque


# =============================================================================
# Transformer Layer Dependency Table
# =============================================================================

TRANSFORMER_DEPENDENCIES = {
    "rms_norm": {
        "depends_on_all": False,
        "same_input_same_output": True,
        "description": "RMSNorm normalization"
    },
    "qkv_projection": {
        "depends_on_all": True,
        "same_input_same_output": True,
        "description": "QKV Projection - token mixing"
    },
    "attention_scores": {
        "depends_on_all": False,
        "same_input_same_output": False,
        "description": "Attention scores + softmax"
    },
    "attention_output": {
        "depends_on_all": True,
        "same_input_same_output": False,
        "description": "Context-dependent mixing"
    },
    "residual_add_1": {
        "depends_on_all": True,
        "same_input_same_output": True,
        "description": "First residual add"
    },
    "mlp_gate_proj": {
        "depends_on_all": True,
        "same_input_same_output": True,
        "description": "MLP gate projection"
    },
    "mlp_up_proj": {
        "depends_on_all": True,
        "same_input_same_output": True,
        "description": "MLP up projection"
    },
    "silu_activation": {
        "depends_on_all": False,
        "same_input_same_output": True,
        "description": "SiLU non-linearity"
    },
    "mlp_down_proj": {
        "depends_on_all": True,
        "same_input_same_output": False,
        "description": "MLP down projection"
    },
    "residual_add_2": {
        "depends_on_all": True,
        "same_input_same_output": True,
        "description": "Second residual add"
    },
    "final_residual": {
        "depends_on_all": True,
        "same_input_same_output": "partial",
        "description": "Final residual connection"
    }
}


# =============================================================================
# Computation Order Definition
# =============================================================================

COMPUTATION_ORDER = [
    ("rms_norm", ["input"]),
    ("qkv_projection", ["rms_norm"]),
    ("attention_scores", ["qkv_projection"]),
    ("attention_output", ["attention_scores"]),
    ("residual_add_1", ["input", "attention_output"]),
    ("rms_norm_2", ["residual_add_1"]),
    ("mlp_gate_proj", ["rms_norm_2"]),
    ("mlp_up_proj", ["rms_norm_2"]),
    ("silu_activation", ["mlp_gate_proj"]),
    ("mlp_down_proj", ["silu_activation", "mlp_up_proj"]),
    ("residual_add_2", ["residual_add_1", "mlp_down_proj"]),
    ("output", ["residual_add_2"])
]


# =============================================================================
# Parallel Execution Analysis
# =============================================================================

def build_dependency_graph():
    """Build dependency graph for computation."""
    graph = defaultdict(list)
    reverse_graph = defaultdict(list)

    for component, dependencies in COMPUTATION_ORDER:
        for dep in dependencies:
            graph[dep].append(component)
            reverse_graph[component].append(dep)

    return graph, reverse_graph


def compute_parallel_stages():
    """
    Compute parallel execution stages.

    Returns list of stages, where each stage contains operations
    that can be executed in parallel.

    Returns:
        List of lists of operation names
    """
    stages = []
    remaining = set(comp for comp, _ in COMPUTATION_ORDER)
    completed = set()

    while remaining:
        ready = []
        for component, deps in COMPUTATION_ORDER:
            if component in remaining and all(d in completed for d in deps):
                ready.append(component)

        if not ready:
            break

        stages.append(ready)
        for comp in ready:
            remaining.remove(comp)
            completed.add(comp)

    return stages


def print_dependency_analysis():
    """Print the full dependency analysis."""
    print("=" * 70)
    print("TRANSFORMER LAYER DEPENDENCY ANALYSIS")
    print("=" * 70)

    # Component dependencies table
    print("\n1. COMPONENT DEPENDENCIES")
    print("-" * 70)
    print(f"{'Component':<25} {'Depends All':<12} {'Deterministic':<14} Description")
    print("-" * 70)

    for name, info in TRANSFORMER_DEPENDENCIES.items():
        depends = "Yes" if info["depends_on_all"] else "No"
        det = str(info["same_input_same_output"])
        print(f"{name:<25} {depends:<12} {det:<14} {info['description']}")

    # Computation order
    print("\n2. COMPUTATION ORDER")
    print("-" * 70)
    for i, (component, deps) in enumerate(COMPUTATION_ORDER, 1):
        dep_str = ", ".join(deps) if deps else "(input)"
        print(f"  {i:2d}. {component:<25} depends on: {dep_str}")

    # Parallel stages
    print("\n3. PARALLEL EXECUTION STAGES")
    print("-" * 70)
    stages = compute_parallel_stages()
    for i, stage in enumerate(stages, 1):
        print(f"  Stage {i}: {', '.join(stage)}")


if __name__ == "__main__":
    print_dependency_analysis()
