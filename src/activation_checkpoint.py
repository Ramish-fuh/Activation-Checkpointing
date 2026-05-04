import copy
import inspect

import torch
import torch.nn as nn
import torch.fx as fx
from typing import Any, Dict, Iterable, List, Optional, Tuple
from torch.fx.experimental.proxy_tensor import make_fx
from torch._functorch.partitioners import _extract_graph_with_inputs_outputs
from graph_tracer import SEPFunction
from mu_two_policy import CheckpointPlan


# We define a custom function that takes in two weight matrices that require
# gradients to be computed and an input data matrix. The function returns the
# gradients of the weight matrices with respect to the loss (sum in our
# example). NOTE: The custom function mimics a simple two layer liner neural
# network with relu activation functions and a sum loss function.
def custom_fn(w1: torch.Tensor, w2: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
    z = torch.mm(w1, x)
    z = nn.functional.relu(z)
    z = torch.mm(z, w2)
    z = nn.functional.relu(z)
    z = z.sum()
    z = SEPFunction.apply(z)
    z.backward()
    return w1.grad, w2.grad


def replace_subsequent_uses_of(
    graph: fx.Graph, old_node: fx.Node, new_node: fx.Node
) -> None:
    old_node_users = set(old_node.users)
    for node in reversed(graph.nodes):
        if node == new_node:
            break
        if node in old_node_users:
            node.replace_input_with(old_node, new_node)


def remove_detach_nodes(gm: fx.GraphModule) -> fx.GraphModule:
    for node in gm.graph.nodes:
        if node.target == torch.ops.aten.detach.default:
            input_node = node.all_input_nodes[0]
            node.replace_all_uses_with(input_node)
            if len(node.users) == 0:
                gm.graph.erase_node(node)
    gm.graph.lint()
    gm.recompile()
    return gm


def get_name_to_node_map(gm: fx.GraphModule) -> Dict[str, fx.Node]:
    name_to_node = {}
    for node in gm.graph.nodes:
        name_to_node[node.name] = node
    return name_to_node


def extract_recompute_subgraph(
    joint_graph: fx.Graph,
    inputs: List[fx.Node],
    outputs: List[fx.Node],
) -> fx.Graph:
    """Call PyTorch's private extractor across supported signatures."""
    kwargs = {
        "joint_graph": joint_graph,
        "inputs": inputs,
        "outputs": outputs,
    }
    params = inspect.signature(_extract_graph_with_inputs_outputs).parameters
    if "outputs_descs" in params:
        kwargs["outputs_descs"] = [None] * len(outputs)
    if "subgraph" in params:
        kwargs["subgraph"] = "forward"

    if len(kwargs) > 3:
        return _extract_graph_with_inputs_outputs(**kwargs)
    return _extract_graph_with_inputs_outputs(joint_graph, inputs, outputs)


def apply_checkpoint_plan(gm: fx.GraphModule, plan: CheckpointPlan) -> fx.GraphModule:
    """Apply a policy-generated checkpoint plan to the joint fwd+bwd graph.

    The plan is expected to contain recompute targets, insertion points, and
    boundary inputs for each target node.
    """
    name_to_node = get_name_to_node_map(gm)
    graph_nodes = list(gm.graph.nodes)
    graph_index = {node: idx for idx, node in enumerate(graph_nodes)}

    def mapped_index(plan_node: fx.Node) -> int:
        mapped = name_to_node.get(plan_node.name)
        return graph_index.get(mapped, 10**9)

    ordered_targets: List[fx.Node] = sorted(
        list(plan.recompute_nodes),
        key=lambda n: mapped_index(plan.first_backward_use.get(n, n)),
    )

    for target in ordered_targets:
        target_name = target.name
        if target_name not in name_to_node:
            raise ValueError(f"Recompute target {target_name} is not in the graph")
        if target not in plan.first_backward_use:
            raise ValueError(f"Recompute target {target_name} has no first backward use")
        if target not in plan.required_recompute_inputs:
            raise ValueError(f"Recompute target {target_name} has no boundary inputs")

        target_node = name_to_node[target_name]
        first_back_access = name_to_node.get(plan.first_backward_use[target].name)
        if first_back_access is None:
            raise ValueError(
                f"First backward use {plan.first_backward_use[target].name} "
                f"for {target_name} is not in the graph"
            )

        required_inputs = []
        missing_inputs = []
        for inp in plan.required_recompute_inputs[target]:
            mapped = name_to_node.get(inp.name)
            if mapped is None:
                missing_inputs.append(inp.name)
                continue
            required_inputs.append(mapped)
        if missing_inputs:
            raise ValueError(
                f"Boundary inputs for {target_name} are not in the graph: "
                + ", ".join(missing_inputs[:10])
            )
        if not required_inputs:
            raise ValueError(f"Recompute target {target_name} has empty boundary inputs")

        recompute_subgraph = extract_recompute_subgraph(
            gm.graph,
            required_inputs,
            [target_node],
        )

        copy_env = dict(name_to_node)
        copied_target = False
        with gm.graph.inserting_before(first_back_access):
            for n in recompute_subgraph.nodes:
                if n.op == "placeholder" or n.op == "output":
                    continue
                new_node = gm.graph.node_copy(
                    n, arg_transform=lambda arg: copy_env[arg.name]
                )
                if n.name == target_name:
                    replace_subsequent_uses_of(
                        gm.graph,
                        old_node=target_node,
                        new_node=new_node,
                    )
                    copied_target = True
                copy_env[n.name] = new_node

        if not copied_target:
            raise ValueError(f"Recompute subgraph did not produce target {target_name}")

        gm.graph.lint()
        gm.recompile()

    return gm


def clone_graph_inputs(args: Iterable[Any]) -> List[Any]:
    """Clone tensor inputs so rewrite smoke checks do not mutate live state."""
    cloned_args: List[Any] = []
    for arg in args:
        if isinstance(arg, torch.Tensor):
            cloned = arg.detach().clone(memory_format=torch.preserve_format)
            if arg.requires_grad and (cloned.is_floating_point() or cloned.is_complex()):
                cloned.requires_grad_(True)
            cloned_args.append(cloned)
        else:
            cloned_args.append(arg)
    return cloned_args


def _rng_state() -> Tuple[torch.Tensor, List[torch.Tensor]]:
    cuda_states: List[torch.Tensor] = []
    if torch.cuda.is_available():
        cuda_states = torch.cuda.get_rng_state_all()
    return torch.random.get_rng_state(), cuda_states


def _restore_rng_state(state: Tuple[torch.Tensor, List[torch.Tensor]]) -> None:
    cpu_state, cuda_states = state
    torch.random.set_rng_state(cpu_state)
    if cuda_states and torch.cuda.is_available():
        torch.cuda.set_rng_state_all(cuda_states)


def _outputs_close(
    expected: Any,
    actual: Any,
    *,
    rtol: float,
    atol: float,
    path: str = "output",
) -> Tuple[bool, str]:
    if isinstance(expected, torch.Tensor) and isinstance(actual, torch.Tensor):
        if expected.shape != actual.shape:
            return False, f"{path} shape mismatch: {expected.shape} != {actual.shape}"
        if expected.dtype != actual.dtype:
            return False, f"{path} dtype mismatch: {expected.dtype} != {actual.dtype}"
        if expected.is_floating_point() or expected.is_complex():
            if not torch.allclose(expected, actual, rtol=rtol, atol=atol, equal_nan=True):
                diff = (expected - actual).abs().max().item()
                return False, f"{path} values differ; max abs diff={diff}"
        elif not torch.equal(expected, actual):
            return False, f"{path} values differ"
        return True, ""

    if isinstance(expected, (list, tuple)) and isinstance(actual, type(expected)):
        if len(expected) != len(actual):
            return False, f"{path} length mismatch: {len(expected)} != {len(actual)}"
        for idx, (expected_item, actual_item) in enumerate(zip(expected, actual)):
            ok, error = _outputs_close(
                expected_item,
                actual_item,
                rtol=rtol,
                atol=atol,
                path=f"{path}[{idx}]",
            )
            if not ok:
                return ok, error
        return True, ""

    if isinstance(expected, dict) and isinstance(actual, dict):
        if expected.keys() != actual.keys():
            return False, f"{path} keys differ"
        for key in expected:
            ok, error = _outputs_close(
                expected[key],
                actual[key],
                rtol=rtol,
                atol=atol,
                path=f"{path}[{key!r}]",
            )
            if not ok:
                return ok, error
        return True, ""

    if expected != actual:
        return False, f"{path} differs: {expected!r} != {actual!r}"
    return True, ""


def smoke_check_graph(
    gm: fx.GraphModule,
    args: Iterable[Any],
    reference_gm: Optional[fx.GraphModule] = None,
    *,
    rtol: float = 1e-4,
    atol: float = 1e-5,
) -> Tuple[bool, str]:
    """Run a transformed graph once on cloned inputs.

    This is deliberately a smoke test, not a numerical proof. It catches the
    common graph-rewrite failures before the compiled benchmark starts replaying
    the modified graph.
    """
    try:
        if reference_gm is not None:
            state = _rng_state()
            with torch.no_grad():
                expected = reference_gm(*clone_graph_inputs(args))
            _restore_rng_state(state)
            with torch.no_grad():
                actual = gm(*clone_graph_inputs(args))
            return _outputs_close(expected, actual, rtol=rtol, atol=atol)

        with torch.no_grad():
            gm(*clone_graph_inputs(args))
    except Exception as exc:
        return False, repr(exc)
    return True, ""


def first_backward_user(gm: fx.GraphModule, target: fx.Node) -> fx.Node:
    """Return the first backward-region user of ``target`` in graph order."""
    nodes = list(gm.graph.nodes)
    node_index = {node: idx for idx, node in enumerate(nodes)}
    sep_backward = next(
        (
            node
            for node in nodes
            if node.target is torch.ops.separator.sep_backward.default
        ),
        None,
    )
    if sep_backward is None:
        raise ValueError("sep_backward marker not found")

    sep_backward_idx = node_index[sep_backward]
    users = [
        user
        for user in target.users
        if node_index[user] >= sep_backward_idx
        and user.target is not torch.ops.separator.sep_backward.default
    ]
    if not users:
        raise ValueError(f"{target.name} has no backward-region users")
    return min(users, key=lambda node: node_index[node])


def require_named_node(name_to_node: Dict[str, fx.Node], *names: str) -> fx.Node:
    """Return the first available node name, or raise a readable error."""
    for name in names:
        if name in name_to_node:
            return name_to_node[name]
    raise ValueError("Missing expected graph node; tried: " + ", ".join(names))


def activation_checkpointing(gm: fx.GraphModule) -> fx.GraphModule:
    # NOTE: You need to create the function for your project and call it inside
    # the graph_transformation function after performing graph profiling.

    # In this example we are going to recompute one of the relu activations for the
    # backward pass instead of saving it. We know from our custom function
    # that we have 2 intermeidate nodes: ['relu', 'relu_1']

    # So the intermediate node to recompute is: ['relu'] and
    # intermediate nodes to checkpoint (retain) are: ['relu_1']

    # Nodes required to recompute 'relu' are ['w1_1', 'x_1']
    # First back use is at node 't'

    # NOTE: For your project, you will use GraphProfiler to identify the
    # intermediate nodes, their first back access, last forward access and
    # then MuTWO's algorithm to select the intermediate 'nodes_to_recompute' and
    # checkpoint (retain). The 'nodes_required_to_recompute' any of the
    # intermediate nodes MUST be a subset of the placeholder nodes and the
    # intermediate nodes that are checkpointed.

    name_to_node = get_name_to_node_map(gm)
    first_back_access = name_to_node["t"]
    node_to_recompute = [name_to_node["relu"]]
    node_to_recompute_names = ["relu"]
    nodes_required_to_recompute = [name_to_node["w1_1"], name_to_node["x_1"]]

    # NOTE: we cannot directly use 'mm' to recompute 'relu' since 'mm' is not an
    # intermediate node that is retained (checkpointed).

    # Obtain a sub-graph that recomputes the required nodes
    recompute_subgraph = extract_recompute_subgraph(
        gm.graph,
        nodes_required_to_recompute,
        node_to_recompute,
    )
    print("Extracted recomputation sub-graph: ")
    recompute_subgraph.print_tabular()

    # Insert the nodes of the new sub-graph in the old graph before the first
    # backward access of the node to be recomputed.
    with gm.graph.inserting_before(first_back_access):
        for n in recompute_subgraph.nodes:
            if n.op == "placeholder" or n.op == "output":
                continue
            # Copy the nodes of the new sub-graph to old graph and transform its
            # inputs to match the old-graph inputs. The arg_transform function
            # will pass the input arguments of the new node and will expect a
            # mapping to the nodes of the old graph.
            new_node = gm.graph.node_copy(
                n, arg_transform=lambda arg: name_to_node[arg.name]
            )

            if n.name in node_to_recompute_names:
                old_node = name_to_node[n.name]
                # Replace all the uses of the old node with new recomputation node
                replace_subsequent_uses_of(
                    gm.graph, old_node=old_node, new_node=new_node
                )
            # Add the new node to our name to node mapping
            name_to_node[n.name] = new_node

    gm.graph.lint()
    gm.recompile()
    return gm


if __name__ == "__main__":
    # Create two weight matrices that require gradients and one input data matrix
    device = "cuda" if torch.cuda.is_available() else "cpu"
    w1 = torch.randn(16, 16, device=device, requires_grad=True)
    w2 = torch.randn(32, 8, device=device, requires_grad=True)
    x = torch.randn(16, 32, device=device)

    # Create a graph module by tracing the the custom function with the given inputs
    graph_module = make_fx(custom_fn)(w1, w2, x)
    graph_module = remove_detach_nodes(graph_module)
    print("Original graph of custom fn (fwd+bwd): ")
    graph_module.graph.print_tabular()

    name_to_node = get_name_to_node_map(graph_module)
    relu_node = require_named_node(name_to_node, "relu")
    first_relu_bw = name_to_node.get("t") or first_backward_user(graph_module, relu_node)
    demo_plan = CheckpointPlan(
        retained_nodes={require_named_node(name_to_node, "relu_1")},
        recompute_nodes={relu_node},
        first_backward_use={relu_node: first_relu_bw},
        required_recompute_inputs={
            relu_node: {
                require_named_node(name_to_node, "w1_1", "w1"),
                require_named_node(name_to_node, "x_1", "x"),
            }
        },
    )

    # Apply the same plan-driven checkpointing helper used by benchmarks.py.
    baseline_graph_module = copy.deepcopy(graph_module)
    new_graph_module = apply_checkpoint_plan(copy.deepcopy(graph_module), demo_plan)
    print("Modified graph of custom fn (fwd+bwd+activation_checkpointing): ")
    new_graph_module.graph.print_tabular()

    # Verify that gradients produced with activation checkpointing equal the
    # ones obtained earlier with no optimization. Use cloned inputs so each
    # graph sees the same initial tensors.
    with torch.no_grad():
        old_grads = baseline_graph_module(*clone_graph_inputs((w1, w2, x)))
        new_grads = new_graph_module(*clone_graph_inputs((w1, w2, x)))

    print("Result verification")
    for old_grad, new_grad in zip(old_grads, new_grads):
        print(torch.allclose(old_grad, new_grad))
