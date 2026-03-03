# ============================================================================
# graph_tracer.py
#
# Traces a full PyTorch training step (forward + backward + optimizer)
# into a single torch.fx GraphModule via make_fx.  Defines sep /
# sep_backward marker ops for region boundary detection and provides
# the compile() decorator for one-shot tracing + graph replay.
# ============================================================================

from contextlib import contextmanager, nullcontext
from copy import copy
from dataclasses import dataclass
from functools import partial, wraps
from typing import Any, Callable, Dict, List, Optional, Union
from utils import SPMD_DECOMP_TABLE

import torch
import torch.distributed._functional_collectives  # side-effect: registers collective ops
import torch.nn as nn
import torch.optim as optim
import torch.utils._pytree as pytree
from torch import fx
from torch._subclasses.fake_tensor import FakeTensorMode
from torch.distributed._functional_collectives import all_reduce
from torch.distributed.tensor import DTensor
from torch.distributed.tensor._op_schema import OpSchema, OutputSharding
from torch.distributed._tensor.placement_types import DTensorSpec
from torch.fx.experimental.proxy_tensor import make_fx
from torch.fx.graph import CodeGen, _PyTreeCodeGen, _PyTreeInfo
from torch.nn.utils import stateless
from torch.utils.hooks import RemovableHandle


# ---------------------------------------------------------------------------
# Separator ops — identity functions registered as custom torch library ops.
# sep(x) marks the start of the forward pass; sep_backward(x) marks the
# start of the backward pass.  They are no-ops at runtime but appear as
# named nodes in the fx graph for region boundary detection.
# ---------------------------------------------------------------------------

def sep(x: torch.Tensor) -> torch.Tensor:
    """Identity op marking the beginning of the forward pass."""
    return x


def sep_backward(grad: torch.Tensor) -> torch.Tensor:
    """Identity op marking the beginning of the backward pass."""
    return grad


# Register sep and sep_backward as first-class torch library ops so they
# survive make_fx tracing and show up as named nodes in the graph.
separator_lib = torch.library.Library("separator", "DEF")
separator_lib.define("sep(Tensor x) -> Tensor")
separator_lib.impl("sep", sep, "CompositeExplicitAutograd")
separator_lib.define("sep_backward(Tensor x) -> Tensor")
separator_lib.impl("sep_backward", sep_backward, "CompositeExplicitAutograd")


# ---------------------------------------------------------------------------
# DTensor sharding propagation rules for the separator ops.
# Both are identities, so the output sharding always matches the input.
# ---------------------------------------------------------------------------

def _identity_prop_rule(op_schema: OpSchema) -> OutputSharding:
    """Return the same DTensorSpec (mesh + placements) as the single input."""
    (x,) = op_schema.args_schema
    assert isinstance(x, DTensorSpec), f"expecting DTensorSpec but got {x}"
    return OutputSharding(output_spec=DTensorSpec(x.mesh, x.placements))

def _prop_sepm(op_schema: OpSchema) -> OutputSharding:
    """Sharding rule for separator.sep – delegates to identity rule."""
    return _identity_prop_rule(op_schema)

def _prop_sepm_backward(op_schema: OpSchema) -> OutputSharding:
    """Sharding rule for separator.sep_backward – delegates to identity rule."""
    return _identity_prop_rule(op_schema)

# Register the sharding rules so DTensor can propagate placements through sep ops.
DTensor._op_dispatcher.sharding_propagator.register_sharding_prop_rule(torch.ops.separator.sep.default, _prop_sepm)
DTensor._op_dispatcher.sharding_propagator.register_sharding_prop_rule(torch.ops.separator.sep_backward.default, _prop_sepm_backward)



class SEPFunction(torch.autograd.Function):
    """Autograd wrapper that inserts sep in the forward pass and
    sep_backward in the backward pass.  Calling SEPFunction.apply(x)
    around a tensor ensures that both boundary markers appear in the
    traced graph after make_fx records the full forward+backward."""

    @staticmethod
    def forward(ctx: Any, x: torch.Tensor) -> torch.Tensor:
        """Forward: call separator.sep (identity) to plant a marker node."""
        return torch.ops.separator.sep(x)

    @staticmethod
    def backward(ctx: Any, grad_x: torch.Tensor) -> torch.Tensor:
        """Backward: call separator.sep_backward (identity) to plant a marker node."""
        return torch.ops.separator.sep_backward(grad_x)


# ---------------------------------------------------------------------------
# Dummy tag_grad op — attached via backward hooks to label gradient
# tensors during tracing.  Cleaned up (erased) in _compile() afterward.
# ---------------------------------------------------------------------------
_spmd_lib_def = torch.library.Library("dummy", "DEF")
_spmd_lib_def.define("tag_grad(Tensor self) -> Tensor")

_spmd_lib_impl = torch.library.Library("dummy", "IMPL")
_spmd_lib_impl.impl("tag_grad", lambda x: x, "CompositeExplicitAutograd")


class _PyTreeCodeGenOutputsOnly(_PyTreeCodeGen):
    """Custom code-gen that only applies pytree flattening/unflattening to the
    *outputs* of the graph module, leaving the inputs as a plain flat list.
    This is used by _to_caller_flattened_graph_module so the caller can pass
    pre-flattened tensors directly instead of wrapping them in a tree."""

    # pyre-ignore[3]
    def process_inputs(self, *args: Any) -> Any:
        # No-op: accept inputs as-is (already flattened by the caller).
        return args

    # pyre-ignore[2, 3]
    def gen_fn_def(self, free_vars, maybe_return_annotation):
        # Fall back to the base CodeGen (not _PyTreeCodeGen) for the function
        # signature, since we no longer expect a tree-structured input.
        return CodeGen.gen_fn_def(self, free_vars, maybe_return_annotation)


def _to_caller_flattened_graph_module(gm: fx.GraphModule) -> fx.GraphModule:
    """Reconfigure *gm* so that its inputs are expected as a flat list of
    tensors (params, buffers, optimizer states, runtime args) instead of
    the original nested pytree structure.  The output spec is preserved.

    This is necessary because _compile() merges all state into one flat
    list before calling gm; without this conversion the graph would try
    to unflatten the inputs using the original tree structure and fail.
    """
    # pyre-ignore[16]
    gm._graph._codegen = _PyTreeCodeGenOutputsOnly(
        pytree_info=_PyTreeInfo(
            # pyre-ignore[6]
            orig_args=None,  # type: ignore[arg-type]
            # pyre-ignore[6]
            in_spec=None,  # type: ignore[arg-type]
            # pyre-ignore[16]
            out_spec=gm._graph._codegen.pytree_info.out_spec,
        )
    )
    gm.graph.eliminate_dead_code()
    gm.recompile()
    return gm


@contextmanager
def gradients_tagging(params: Dict[str, nn.Parameter]):
    """Context manager that registers backward hooks on every parameter.

    Each hook calls the dummy.tag_grad identity op on the gradient tensor.
    During tracing with make_fx, this causes a tag_grad node to appear in
    the graph right after the gradient is produced, letting downstream code
    (profiler, SPMD) classify those tensors as GRAD.

    The hooks are removed once the context exits (after tracing is done).
    The corresponding tag_grad nodes are erased later in _compile().
    """
    tagging_hooks: List[RemovableHandle] = []
    try:
        for p in params.values():
            h = p.register_hook(lambda grad: torch.ops.dummy.tag_grad(grad))
            tagging_hooks.append(h)
        yield
    finally:
        for h in tagging_hooks:
            h.remove()


@contextmanager
def _rematerialize_optimizer(
    opt: optim.Optimizer,
    named_states: Dict[str, Any],
    params: Dict[str, nn.Parameter],
):
    """Temporarily swap the optimizer's internal state and parameter
    references with the proxy/fake tensors used during tracing.

    This is required because make_fx replaces real tensors with proxies.
    The optimizer still needs to find its momentum buffers etc. via the
    same keys (Parameters), so we redirect opt.state and param_groups to
    use the proxy versions.  After tracing, the original state and params
    are restored.

    NOTE: Currently only supports a single parameter group.
    """
    assert opt is not None

    # Temporarily point opt.state entries to the proxy named_states.
    orig_states = copy(opt.state)
    for n in named_states:
        opt.state[params[n]] = named_states[n]  # type: ignore[index]

    # Swap the param list in the (single) param_group to the proxy params.
    param_group = opt.param_groups[0]
    orig_params = param_group["params"]
    param_group["params"] = params.values()

    try:
        yield
    finally:
        # Restore originals so the real optimizer is unaffected.
        param_group["params"] = orig_params
        opt.state = orig_states


@contextmanager
def _enable_compile():
    """Monkey-patch torch._utils.is_compiling() to return True.

    PyTorch's optimizer skips certain graph-unfriendly code paths when it
    detects a compile context.  By forcing is_compiling to return True we
    ensure the optimizer step is fully traceable by make_fx, so the
    resulting graph includes weight update ops (e.g. addcdiv for Adam).
    The original function is restored when the context exits.
    """
    def f_true():
        return True

    orig_is_compiling_code = torch._utils.is_compiling.__code__
    torch._utils.is_compiling.__code__ = f_true.__code__
    try:
        yield
    finally:
        torch._utils.is_compiling.__code__ = orig_is_compiling_code


@dataclass
class _CompiledResult:
    """Container returned by _compile() holding everything needed to
    replay a traced training step:  the fx GraphModule, the original
    nn.Module, the optimizer, and the flattened list of all stateful
    tensors (params + buffers + optimizer states)."""
    gm: fx.GraphModule
    mod: nn.Module
    opt: Optional[torch.optim.Optimizer]
    flat_state: List[torch.Tensor]


def _compile(func: Callable, *args: Any, **kwargs: Any):
    """Trace a full training step into a single fx.GraphModule.

    High-level flow:
      1. Extract the nn.Module and Optimizer from the function arguments.
      2. Lift all model parameters, buffers, and optimizer states into
         explicit function arguments ("stateless" form) so make_fx can
         record every operation performed on them.
      3. Convert real tensors to FakeTensors (meta-only, no data) for
         lightweight tracing.
      4. Call make_fx to symbolically trace the training step, producing
         an fx.GraphModule whose graph contains forward, backward, and
         optimizer update ops.
      5. Clean up: remove bookkeeping-only nodes (aten.detach and
         dummy.tag_grad) that were needed only during tracing.
      6. Flatten the graph's input contract so the caller can pass a
         single flat tensor list.

    Returns a _CompiledResult with the traced graph + all state.
    """

    # ---- Step 1: Find the nn.Module and Optimizer in the arguments ----
    mod, opt = None, None
    for arg in pytree.tree_flatten(list(args) + list(kwargs.values()))[0]:
        if isinstance(arg, nn.Module):
            assert mod is None, "Only support single nn.Module for now"
            mod = arg
        if isinstance(arg, optim.Optimizer):
            assert opt is None, "Only support single Optimizer for now"
            opt = arg
    assert mod is not None, "Couldn't find nn.Module instances from the arguments."

    # ---- Step 2: Lift params, buffers, optimizer states as function args ----
    params = dict(mod.named_parameters(remove_duplicate=False))
    buffers = dict(mod.named_buffers(remove_duplicate=False))

    # Build a dict mapping param name -> optimizer state dict for that param.
    # We key by name (string) rather than by Parameter object because during
    # tracing the Parameter objects are replaced with proxies.
    named_states: Dict[str, nn.Parameter] = {}
    for n, p in params.items():
        if p in opt.state:
            named_states[n] = opt.state[p]

    def stateless_func(
        func: Callable,
        params: Dict[str, nn.Parameter],
        buffers: Dict[str, torch.Tensor],
        named_states: Dict[str, nn.Parameter],
        args: Any,
        kwargs: Any,
    ):
        """Wrapper that replaces the module's parameters/buffers with the
        traced proxy tensors and runs the user's train_step function.
        Returns (train_step output, updated params, updated opt states)."""
        with stateless._reparametrize_module(
            mod, {**params, **buffers}
        ), _rematerialize_optimizer(
            opt, named_states, params
        ) if opt else nullcontext():
            # Tag every parameter's gradient so it is identifiable in the graph.
            with gradients_tagging(params):
                ret = func(*args, **kwargs)

            # Return updated parameters and optimizer states alongside the
            # original return value so they become graph outputs.
            return ret, list(mod.parameters()), list(named_states.values())

    # ---- Step 3: Convert all tensor arguments to FakeTensors ----
    # FakeTensors carry shape/dtype/device metadata but hold no data,
    # making tracing fast and memory-free.
    tracing_mode = "fake"
    fake_mode = FakeTensorMode()

    def _get_fake_args(arg: torch.Tensor) -> torch.Tensor:
        return fake_mode.from_tensor(arg)

    args = pytree.tree_map_only(torch.Tensor, _get_fake_args, args)
    kwargs = pytree.tree_map_only(torch.Tensor, _get_fake_args, kwargs)

    # ---- Step 4: Trace forward + backward + optimizer via make_fx ----
    with _enable_compile(), torch.autograd.detect_anomaly(check_nan=False):
        gm = make_fx(
            partial(stateless_func, func),
            tracing_mode=tracing_mode,
            decomposition_table=SPMD_DECOMP_TABLE,
            _allow_non_fake_inputs=False,
        )(params, buffers, named_states, args, kwargs)

    # ---- Step 5: Clean up tracing-only nodes ----
    # Flatten all state into a single list to serve as graph inputs.
    params_and_buffers: Dict[str, Union[torch.Tensor, nn.Parameter]] = {
        **params,
        **buffers,
    }
    flat_state, _ = pytree.tree_flatten([params_and_buffers, named_states])

    for node in gm.graph.nodes:
        # Remove aten.detach nodes introduced by autograd bookkeeping.
        if node.target == torch.ops.aten.detach.default:
            input_node = node.all_input_nodes[0]
            node.replace_all_uses_with(input_node)
            if len(node.users) == 0:
                gm.graph.erase_node(node)
        # Remove dummy.tag_grad nodes – they were only needed to label
        # gradient tensors during tracing; no longer necessary.
        if node.target == torch.ops.dummy.tag_grad.default:
            grad_node = node.all_input_nodes[0]
            node.replace_all_uses_with(grad_node)
            if len(node.users) == 0:
                gm.graph.erase_node(node)

    # ---- Step 6: Flatten the graph's input contract ----
    gm = _to_caller_flattened_graph_module(gm)

    return _CompiledResult(gm, mod, opt, flat_state)


# Key used to cache the _CompiledResult on the wrapper function so that
# tracing happens only once (on the first call) and subsequent calls
# simply replay the graph.
COMPILED_OBJECT_KEY = "_compiled_obj"


def compile(func: Callable, gm_transformation: Callable):
    """Public decorator that traces *func* (a training step) on the first
    call and replays the graph on all subsequent calls.

    Args:
        func: A training step function (forward + loss + backward + opt.step)
              that takes an nn.Module and Optimizer among its arguments.
        gm_transformation: A callable(gm, flat_inputs) -> gm applied once
              after tracing to transform the graph (e.g. insert activation
              checkpointing, profiling, etc.).

    Returns:
        A wrapper function with the same signature as *func*.  On the
        first invocation the function is traced via _compile(), the
        gm_transformation is applied, and the graph is executed.  On
        later invocations the cached graph is directly replayed.
    """
    @wraps(func)
    def wrapper(*args, **kwargs):
        first_iter = False
        compiled_obj = wrapper.__dict__.get(COMPILED_OBJECT_KEY, None)

        if compiled_obj is None:
            # First call: trace the entire training step into an fx graph.
            first_iter = True
            compiled_obj = _compile(func, *args, **kwargs)
            wrapper.__dict__[COMPILED_OBJECT_KEY] = compiled_obj

        # Build the flat input list: [params, buffers, opt_states, *args, *kwargs]
        flat_inps = compiled_obj.flat_state + pytree.tree_flatten([args, kwargs])[0]

        if first_iter and gm_transformation:
            # Apply the user's graph transformation (profiling, AC, etc.)
            compiled_obj.gm = gm_transformation(compiled_obj.gm, flat_inps)

        # Execute the traced graph with no_grad since gradients are already
        # baked into the graph as explicit ops.
        with torch.no_grad():
            output = compiled_obj.gm(*flat_inps)[0]

        return output

    return wrapper
