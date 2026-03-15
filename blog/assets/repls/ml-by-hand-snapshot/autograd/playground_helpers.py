from __future__ import annotations

from collections.abc import Callable, Iterable, Mapping

from autograd.backend import xp
from autograd.tensor import Tensor


def _iter_inputs(tensor: Tensor) -> tuple[Tensor, ...]:
    if tensor.creator is None or tensor.creator.tensors is None:
        return ()
    return tuple(inp for inp in tensor.creator.tensors if inp is not None)


def _iter_backward_inputs(tensor: Tensor) -> tuple[Tensor, ...]:
    return tuple(inp for inp in _iter_inputs(tensor) if inp.requires_grad)


def _topological_order(
    root: Tensor,
    *,
    input_iter: Callable[[Tensor], tuple[Tensor, ...]] | None = None,
) -> list[Tensor]:
    order: list[Tensor] = []
    seen: set[int] = set()
    iter_inputs = _iter_inputs if input_iter is None else input_iter

    def walk(node: Tensor) -> None:
        node_id = id(node)
        if node_id in seen:
            return
        seen.add(node_id)
        for inp in iter_inputs(node):
            walk(inp)
        order.append(node)

    walk(root)
    return order


def _scalar_text(tensor: Tensor | None) -> str:
    if tensor is None:
        return "None"
    if tensor.data.shape == ():
        value = xp.to_scalar(tensor.data)
        if isinstance(value, float) and value.is_integer():
            return str(int(value))
        return repr(value)
    if hasattr(tensor.data, "tolist"):
        return repr(tensor.data.tolist())
    return repr(tensor.data)


def grad_value(tensor: Tensor) -> object | None:
    if tensor.grad is None:
        return None
    return xp.to_scalar(tensor.grad.data)


def step_value(tensor: Tensor, learning_rate: float) -> object:
    grad = grad_value(tensor)
    if grad is None:
        return xp.to_scalar(tensor.data)
    return xp.to_scalar(tensor.data) - learning_rate * grad


def build_running_example(
    *,
    x_value: float = 2.0,
    w_value: float = 3.0,
    b_value: float = 1.0,
    y_true_value: float = 8.0,
    x_requires_grad: bool = False,
    w_requires_grad: bool = True,
    b_requires_grad: bool = True,
    y_true_requires_grad: bool = False,
) -> dict[str, object]:
    x = Tensor(x_value, requires_grad=x_requires_grad)
    w = Tensor(w_value, requires_grad=w_requires_grad)
    b = Tensor(b_value, requires_grad=b_requires_grad)
    y_true = Tensor(y_true_value, requires_grad=y_true_requires_grad)

    prod = x * w
    pred = prod + b
    error = pred - y_true
    loss = error ** 2

    names = {
        "x": x,
        "w": w,
        "b": b,
        "y_true": y_true,
        "prod": prod,
        "pred": pred,
        "error": error,
        "loss": loss,
    }

    return {
        "x": x,
        "w": w,
        "b": b,
        "y_true": y_true,
        "prod": prod,
        "pred": pred,
        "error": error,
        "loss": loss,
        "names": names,
    }


def print_running_example_update(
    *,
    learning_rate: float = 0.1,
    y_true_value: float = 8.0,
) -> None:
    example = build_running_example(y_true_value=y_true_value)
    pred = example["pred"]
    loss = example["loss"]
    w = example["w"]
    b = example["b"]
    x = example["x"]
    y_true = example["y_true"]

    print("Before the update:")
    print(f"  pred = {_scalar_text(pred)}")
    print(f"  loss = {_scalar_text(loss)}")

    print()
    print("After backward:")
    try:
        loss.backward()
    except Exception as error:
        print(f"  backward failed: {error}")
        return

    print(f"  w.grad = {grad_value(w)}")
    print(f"  b.grad = {grad_value(b)}")

    w_new = step_value(w, learning_rate)
    b_new = step_value(b, learning_rate)
    pred_after = xp.to_scalar(x.data) * w_new + b_new
    loss_after = (pred_after - xp.to_scalar(y_true.data)) ** 2

    print()
    print("After one gradient step:")
    print(f"  learning_rate = {learning_rate}")
    print(f"  w_new = {w_new}")
    print(f"  b_new = {b_new}")
    print(f"  pred_after = {pred_after}")
    print(f"  loss_after = {loss_after}")


def _id_names(named_tensors: Mapping[str, Tensor] | None) -> dict[int, str]:
    if named_tensors is None:
        return {}
    return {id(tensor): name for name, tensor in named_tensors.items()}


def _label_for(tensor: Tensor, id_names: Mapping[int, str]) -> str:
    return id_names.get(id(tensor), f"anon({_scalar_text(tensor)})")


def _iter_display_inputs(
    tensor: Tensor,
    id_names: Mapping[int, str],
) -> tuple[Tensor, ...]:
    if not id_names:
        return _iter_inputs(tensor)

    visible: list[Tensor] = []

    def collect(node: Tensor) -> None:
        if node.requires_grad or id(node) in id_names:
            visible.append(node)
            return
        for inp in _iter_inputs(node):
            collect(inp)

    for inp in _iter_inputs(tensor):
        collect(inp)

    return tuple(visible)


def _named_nodes(
    root: Tensor,
    named_tensors: Mapping[str, Tensor] | None,
) -> list[tuple[str, Tensor]]:
    if named_tensors is None:
        return []
    id_names = _id_names(named_tensors)
    return [
        (id_names[id(node)], node) for node in _topological_order(root) if id(node) in id_names
    ]


def _print_tree(
    node: Tensor,
    id_names: Mapping[int, str],
    *,
    prefix: str = "",
    is_last: bool = True,
    seen: set[int] | None = None,
) -> None:
    seen = seen if seen is not None else set()
    marker = "└── " if is_last else "├── "
    creator = node.creator.__class__.__name__ if node.creator else "leaf"
    print(f"{prefix}{marker}{_label_for(node, id_names)}: data={_scalar_text(node)}, creator={creator}")
    node_id = id(node)
    if node_id in seen:
        loop_prefix = prefix + ("    " if is_last else "│   ")
        print(f"{loop_prefix}└── <already shown>")
        return
    seen.add(node_id)
    inputs = _iter_display_inputs(node, id_names)
    next_prefix = prefix + ("    " if is_last else "│   ")
    for index, inp in enumerate(inputs):
        _print_tree(inp, id_names, prefix=next_prefix, is_last=index == len(inputs) - 1, seen=seen)


def print_graph_structure(
    root: Tensor,
    named_tensors: Mapping[str, Tensor] | None = None,
) -> None:
    named_nodes = [(name, node) for name, node in _named_nodes(root, named_tensors) if node.creator is not None]
    id_names = _id_names(named_tensors)
    if named_nodes:
        print("Forward values:")
        for name, node in named_nodes:
            print(f"  {name} = {_scalar_text(node)}")
        print()
        print("Creator links:")
        for name, node in named_nodes:
            print(f"  {name}.creator -> {node.creator.__class__.__name__}")
        print()
    print("Dependency tree:")
    _print_tree(root, id_names)


def print_backward_walk(
    root: Tensor,
    named_tensors: Mapping[str, Tensor] | None = None,
    *,
    show_grads: Iterable[str] = (),
) -> None:
    id_names = _id_names(named_tensors)
    reverse_walk = [
        node
        for node in reversed(_topological_order(root, input_iter=_iter_backward_inputs))
        if node.creator is not None
    ]
    print("Backward visit order:")
    print("  seed loss with grad 1")
    for index, node in enumerate(reverse_walk, start=1):
        inputs = ", ".join(_label_for(inp, id_names) for inp in _iter_backward_inputs(node))
        print(f"  {index}. {_label_for(node, id_names)} [{node.creator.__class__.__name__}] -> {inputs}")
    root.backward()
    named_lookup = dict(named_tensors or ())
    if tuple(show_grads):
        print()
        print("Stored gradients after backward:")
        for name in show_grads:
            tensor = named_lookup.get(name)
            if tensor is None:
                continue
            print(f"  {name}.grad = {_scalar_text(tensor.grad)}")
