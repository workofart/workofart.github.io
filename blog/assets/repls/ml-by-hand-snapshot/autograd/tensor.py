from __future__ import annotations
import logging
from abc import abstractmethod
from typing import Any, List, Optional, Sequence, Tuple, Union, cast
from autograd.backend import ARRAY_TYPE, Array, ArrayLike, xp
logger = logging.getLogger(__name__)

class Function:

    def __init__(self, *tensors: 'Tensor'):
        self.tensors = tensors

    @abstractmethod
    def forward(self, *args: Array, **kwargs: Any) -> Array:
        raise NotImplementedError('Forward pass not implemented for this function')

    @abstractmethod
    def backward(self, grad: 'Tensor') -> Array:
        raise NotImplementedError('Backward pass not implemented for this function')

    @classmethod
    def apply(cls, *tensors: 'Tensor', **kwargs: Any) -> 'Tensor':
        func = cls(*tensors)
        out_data = func.forward(*(inp.data for inp in tensors), **kwargs)
        requires_grad = any((inp.requires_grad for inp in tensors))
        out = Tensor(out_data, creator=func, requires_grad=requires_grad)
        return out

    @staticmethod
    def unbroadcast(grad_arr: Array, to_shape: Tuple[int, ...]) -> Array:
        if grad_arr.shape == to_shape:
            return grad_arr
        while len(grad_arr.shape) > len(to_shape):
            grad_arr = grad_arr.sum(axis=0, keepdims=False)
        for dim in range(len(to_shape)):
            if to_shape[dim] == 1 and grad_arr.shape[dim] != 1:
                grad_arr = grad_arr.sum(axis=dim, keepdims=True)
        return grad_arr

class Tensor:

    def __init__(self, data: ArrayLike, creator: Optional[Function]=None, requires_grad: bool=True):
        self.data = data
        self._grad: Optional['Tensor'] = None
        self.creator = creator
        self._backward = lambda: None
        self.requires_grad = requires_grad

    @property
    def data(self) -> Array:
        return self._data

    @data.setter
    def data(self, value: ArrayLike) -> None:
        if isinstance(value, ARRAY_TYPE):
            if getattr(value, 'dtype', None) == xp.float32:
                self._data = value
                return
        else:
            self._data = xp.array(value, dtype=xp.float32)
            return
        self._data = xp.array(value, dtype=xp.float32)

    @property
    def grad(self) -> Optional['Tensor']:
        if self._grad is not None and (not isinstance(self._grad, Tensor)):
            return Tensor(self._grad, requires_grad=False)
        return self._grad

    @grad.setter
    def grad(self, value: Union['Tensor', ArrayLike, None]) -> None:
        if value is None:
            self._grad = None
            return
        if isinstance(value, Tensor):
            value_data = value.data
        else:
            value_data = xp.array(value)
        if self._grad is None:
            self._grad = Tensor(value_data, requires_grad=False)
        else:
            value_data = xp.broadcast_to(value_data, value_data.shape)
            value_data = xp.array(value_data, dtype=value_data.dtype)
            self._grad.data += value_data

    def view(self, *shape: Union[int, Tuple[int, ...]]) -> 'Tensor':
        if len(shape) == 1 and isinstance(shape[0], tuple):
            new_shape = shape[0]
        else:
            new_shape = cast(Tuple[int, ...], shape)
        return View.apply(self, new_shape=new_shape)

    @staticmethod
    def stack(tensors: List['Tensor'], axis: int=0) -> 'Tensor':
        return Stack.apply(*tensors, axis=axis)

    @staticmethod
    def cat(tensors: List['Tensor'], axis: int=0) -> 'Tensor':
        return Cat.apply(*tensors, axis=axis)

    def __add__(self, other: Union['Tensor', float, int]) -> 'Tensor':
        if not isinstance(other, Tensor):
            other = Tensor(other, requires_grad=False)
        return Add.apply(self, other)

    def __mul__(self, other: Union['Tensor', float, int]) -> 'Tensor':
        if not isinstance(other, Tensor):
            other = Tensor(other, requires_grad=False)
        return Mul.apply(self, other)

    def __matmul__(self, other: Union['Tensor', float, int]) -> 'Tensor':
        if not isinstance(other, Tensor):
            other = Tensor(other, requires_grad=False)
        return Matmul.apply(self, other)

    def __pow__(self, other: Union['Tensor', float, int]) -> 'Tensor':
        if isinstance(other, int) and other >= 0:
            if other == 0:
                return Tensor(xp.ones_like(self.data), requires_grad=False)
            result = self
            for _ in range(other - 1):
                result = result * self
            return result
        if not isinstance(other, Tensor):
            other = Tensor(other, requires_grad=False)
        return Pow.apply(self, other)

    def __iadd__(self, other: Union['Tensor', float, int]) -> 'Tensor':
        if not isinstance(other, Tensor):
            other = Tensor(other)
        broadcast_shape = xp.broadcast_shapes(self.shape, other.shape)
        expanded_other = other.expand(broadcast_shape)
        return IAdd.apply(self, expanded_other)

    def __getitem__(self, idx: Union[int, slice, tuple]) -> 'Tensor':
        return GetItem.apply(self, idx=idx)

    def __setitem__(self, idx: Union[int, slice, tuple], value: Union['Tensor', float, int]) -> 'Tensor':
        if not isinstance(value, Tensor):
            value = Tensor(value, requires_grad=False)
        return SetItem.apply(self, value, idx=idx)

    def sum(self, axis: Optional[Union[int, Tuple[int, ...]]]=None, keepdims: bool=False) -> 'Tensor':
        return Sum.apply(self, axis=axis, keepdims=keepdims)

    def mean(self, axis: Optional[Union[int, Tuple[int, ...]]]=None, keepdims: bool=False) -> 'Tensor':
        return Mean.apply(self, axis=axis, keepdims=keepdims)

    def max(self, axis: Optional[Union[int, Tuple[int, ...]]]=None, keepdims: bool=False) -> 'Tensor':
        return Max.apply(self, axis=axis, keepdims=keepdims)

    def gather(self, index: Any=0) -> 'Tensor':
        return Gather.apply(self, index=index)

    def sqrt(self) -> 'Tensor':
        return Sqrt.apply(self)

    def maximum(self, other: Union['Tensor', float, int]) -> 'Tensor':
        if not isinstance(other, Tensor):
            other = Tensor(other, requires_grad=False)
        return Maximum.apply(self, other)

    def pad(self, pad_width: Union[int, Tuple[int, int], Tuple[int, int, int, int], Tuple[Tuple[int, int], ...]], mode: str='constant', constant_values: Union[int, float]=0) -> 'Tensor':
        return Pad.apply(self, pad_width=pad_width, mode=mode, constant_values=constant_values)

    def forward(self, data: Any) -> None:
        pass

    def backward(self, grad: Optional[Union['Tensor', ArrayLike]]=None) -> None:
        if not self.requires_grad:
            return
        if grad is None:
            grad = Tensor(xp.ones_like(self.data))
        self.grad = grad
        topological_sorted_tensors: list[Tensor] = []
        visited: set[Tensor] = set()
        stack: list[tuple[Tensor, bool]] = [(self, False)]
        while stack:
            node, has_visited_children = stack.pop()
            if node not in visited:
                if not has_visited_children:
                    stack.append((node, True))
                    if node.creator is not None:
                        for p in node.creator.tensors:
                            if p.requires_grad:
                                stack.append((p, False))
                else:
                    visited.add(node)
                    topological_sorted_tensors.append(node)
        for tensor in reversed(topological_sorted_tensors):
            if tensor.creator is not None:
                assert tensor.grad is not None
                grads = tensor.creator.backward(tensor.grad)
                if not isinstance(grads, tuple):
                    grads = (grads,)
                for input_tensor, g in zip(tensor.creator.tensors, grads):
                    if input_tensor is not None and input_tensor.requires_grad and (g is not None):
                        input_tensor._accumulate_grad(g)
                cast(Any, tensor.creator).tensors = None
        for node in topological_sorted_tensors:
            node.creator = None

    @property
    def shape(self) -> Tuple[int, ...]:
        if isinstance(self.data, (int, float)) or not hasattr(self.data, 'shape'):
            return ()
        return self.data.shape

    def reshape(self, *shape: int) -> 'Tensor':
        return Reshape.apply(self, shape=shape)

    def expand(self, *shape: Union[int, Sequence[int]]) -> 'Tensor':
        if len(shape) == 1 and isinstance(shape[0], (tuple, list)):
            shape = tuple(shape[0])
        return Expand.apply(self, shape=shape)

    def permute(self, *dims: int) -> 'Tensor':
        return Permute.apply(self, dims=dims)

    def transpose(self, dim0: int=0, dim1: int=1) -> 'Tensor':
        return Transpose.apply(self, dim0=dim0, dim1=dim1)

    def strided_windows(self, kernel_size: int, stride: int) -> 'Tensor':
        return StridedWindows.apply(self, kernel_size=kernel_size, stride=stride)

    def roll(self, shifts: int, dims: int) -> 'Tensor':
        return Roll.apply(self, shifts=shifts, dims=dims)

    def detach(self) -> 'Tensor':
        return Tensor(self.data, requires_grad=False)

    def item(self) -> Any:
        return xp.to_scalar(self.data)

    def numpy(self) -> Any:
        return xp.to_numpy(self.data)

    @property
    def ndim(self) -> int:
        return len(self.data.shape)

    @property
    def T(self) -> 'Tensor':
        if len(self.data.shape) != 2:
            raise ValueError('T property is only defined for 2D tensors. Use transpose() for higher dimensions.')
        return self.transpose(1, 0)

    def _accumulate_grad(self, grad: Union['Tensor', Array], idx: Optional[Any]=None) -> None:
        if grad is None:
            return
        if not isinstance(grad, Tensor):
            grad = Tensor(grad, requires_grad=False)
        if self._grad is None:
            if idx is not None:
                self._grad = Tensor(xp.zeros_like(self.data), requires_grad=False)
                self._grad.data[idx] = grad.data
            else:
                self._grad = grad
        elif idx is not None:
            self._grad.data[idx] += grad.data
        else:
            self._grad.data += grad.data

    def __radd__(self, other: Union['Tensor', float, int]) -> 'Tensor':
        return self + other

    def __rmul__(self, other: Union['Tensor', float, int]) -> 'Tensor':
        return self * other

    def __sub__(self, other: Union['Tensor', float, int]) -> 'Tensor':
        return self + -other

    def __rsub__(self, other: Union['Tensor', float, int]) -> 'Tensor':
        return other + -self

    def __truediv__(self, other: Union['Tensor', float, int]) -> 'Tensor':
        return self * other ** (-1)

    def __neg__(self) -> 'Tensor':
        return self * -1

    def __repr__(self) -> str:
        return f'Tensor(data={self.data}, grad={self.grad})'

    def __lt__(self, other: Union['Tensor', float, int]) -> Union[Array, bool]:
        return self.data < (other.data if isinstance(other, Tensor) else other)

    def __le__(self, other: Union['Tensor', float, int]) -> Union[Array, bool]:
        return self.data <= (other.data if isinstance(other, Tensor) else other)

    def __gt__(self, other: Union['Tensor', float, int]) -> Union[Array, bool]:
        return self.data > (other.data if isinstance(other, Tensor) else other)

    def __ge__(self, other: Union['Tensor', float, int]) -> Union[Array, bool]:
        return self.data >= (other.data if isinstance(other, Tensor) else other)

    def __eq__(self, other: Union['Tensor', float, int]) -> Union[Array, bool]:
        return self.data == (other.data if isinstance(other, Tensor) else other)

    def __hash__(self) -> int:
        return id(self)

class Add(Function):

    def forward(self, x: Array, y: Array) -> Array:
        self.x_shape = x.shape
        self.y_shape = y.shape
        return x + y

    def backward(self, grad: 'Tensor') -> Tuple[Optional[Array], Optional[Array]]:
        grad_x = grad.data if self.tensors[0].requires_grad else None
        grad_y = grad.data if self.tensors[1].requires_grad else None
        if grad_x is not None:
            grad_x = Function.unbroadcast(grad_x, self.x_shape)
        if grad_y is not None:
            grad_y = Function.unbroadcast(grad_y, self.y_shape)
        return (grad_x, grad_y)

class Mul(Function):

    def forward(self, x: Array, y: Array) -> Array:
        self.x_shape = x.shape
        self.y_shape = y.shape
        return x * y

    def backward(self, grad: 'Tensor') -> Tuple[Optional[Array], Optional[Array]]:
        grad_x = grad.data * self.tensors[1].data if self.tensors[0].requires_grad else None
        grad_y = grad.data * self.tensors[0].data if self.tensors[1].requires_grad else None
        if grad_x is not None:
            grad_x = Function.unbroadcast(grad_x, self.x_shape)
        if grad_y is not None:
            grad_y = Function.unbroadcast(grad_y, self.y_shape)
        return (grad_x, grad_y)

class Pow(Function):

    def forward(self, x: Array, y: Array) -> Array:
        self.x_shape = x.shape
        self.y_shape = y.shape
        return x ** y

    def backward(self, grad: 'Tensor') -> Tuple[Optional[Array], Optional[Array]]:
        x = self.tensors[0]
        y = self.tensors[1]
        grad_x = None
        grad_y = None
        if x.requires_grad:
            grad_x = y.data * x.data ** (y.data - 1) * grad.data
        if y.requires_grad:
            valid_base = x.data > 0
            grad_y = x.data ** y.data * xp.log(xp.abs(x.data)) * grad.data
            grad_y = xp.where(valid_base, grad_y, 0)
        if grad_x is not None:
            grad_x = Function.unbroadcast(grad_x, self.x_shape)
        if grad_y is not None:
            grad_y = Function.unbroadcast(grad_y, self.y_shape)
        return (grad_x, grad_y)

class Matmul(Function):

    def forward(self, x: Array, y: Array) -> Array:
        self.x_shape = x.shape
        self.y_shape = y.shape
        out = xp.matmul(x, y)
        return out

    def backward(self, grad: 'Tensor') -> Tuple[Optional[Array], Optional[Array]]:
        x = self.tensors[0]
        y = self.tensors[1]
        grad_x = grad_y = None
        if x.data.ndim == 1 and y.data.ndim == 1:
            if x.requires_grad:
                grad_x = grad.data * y.data
            if y.requires_grad:
                grad_y = grad.data * x.data
            return (grad_x, grad_y)
        if x.requires_grad:
            y_t = xp.swapaxes(y.data, -1, -2)
            grad_x = xp.matmul(grad.data, y_t)
        if y.requires_grad:
            x_t = xp.swapaxes(x.data, -1, -2)
            raw_grad_y = xp.matmul(x_t, grad.data)
            if y.data.ndim == 2 and raw_grad_y.ndim > 2:
                axes_to_sum = tuple(range(raw_grad_y.ndim - 2))
                grad_y = xp.sum(raw_grad_y, axis=axes_to_sum)
            else:
                grad_y = raw_grad_y
        return (grad_x, grad_y)

class IAdd(Function):

    def forward(self, x: Array, y: Array) -> Array:
        x += y
        return x

    def backward(self, grad: 'Tensor') -> Tuple[Array, Array]:
        return (grad.data, grad.data)

class GetItem(Function):

    def forward(self, x: Array, idx: Any) -> Array:
        self.idx = idx
        return x[idx]

    def backward(self, grad: 'Tensor') -> Array:
        grad_data = grad.data
        out = xp.zeros_like(self.tensors[0].data)
        out = xp.scatter_add(out, self.idx, grad_data)
        return out

class SetItem(Function):

    def forward(self, x: Array, value: Array, idx: Any) -> Array:
        val_data = xp.squeeze(value)
        x[idx] = val_data
        self.idx = idx
        return x

    def backward(self, grad: 'Tensor') -> Array:
        return grad.data[self.idx]

class Sqrt(Function):

    def forward(self, x: Array) -> Array:
        self.x = x
        return xp.sqrt(x)

    def backward(self, grad: 'Tensor') -> Array:
        return grad.data * 0.5 / xp.sqrt(self.x)

class Sum(Function):

    def forward(self, x: Array, axis: Optional[Union[int, Tuple[int, ...]]]=None, keepdims: bool=False) -> Array:
        if not hasattr(x, 'ndim') or x.ndim == 0:
            return x
        self.axis = (axis,) if isinstance(axis, int) else axis
        self.keepdims = keepdims
        self.x_shape = x.shape
        return xp.sum(x, axis=self.axis, keepdims=self.keepdims)

    def backward(self, grad: 'Tensor') -> Array:
        if grad is None:
            return None
        grad_arr = grad.data
        if isinstance(self.axis, int):
            reduce_axes = (self.axis,)
        else:
            reduce_axes = self.axis
        if reduce_axes is None:
            return xp.asarray(xp.broadcast_to(grad_arr, self.x_shape))
        if not self.keepdims:
            for ax in sorted(reduce_axes):
                grad_arr = xp.expand_dims(grad_arr, ax)
        return xp.asarray(xp.broadcast_to(grad_arr, self.x_shape))

class Max(Function):

    def forward(self, x: Array, axis: Optional[Union[int, Tuple[int, ...]]]=None, keepdims: bool=False) -> Array:
        axis = (axis,) if isinstance(axis, int) else axis
        self.axis = axis
        self.keepdims = keepdims
        return xp.max(x, axis=axis, keepdims=keepdims)

    def backward(self, grad: 'Tensor') -> Array:
        x = self.tensors[0]
        max_vals = xp.max(x.data, axis=self.axis, keepdims=True)
        mask = x.data == max_vals
        if self.axis is None or (isinstance(self.axis, tuple) and len(self.axis) > 1):
            count = xp.sum(cast(Any, mask))
            return grad.data * mask / count
        else:
            assert self.axis is not None
            if isinstance(self.axis, tuple):
                assert len(self.axis) == 1
                ax = self.axis[0]
            else:
                ax = self.axis
            cumsum = xp.cumsum(cast(Any, mask), axis=ax)
            first_occur = cumsum == 1
            return grad.data * (mask * first_occur)

class Maximum(Function):

    def forward(self, x: Array, y: Array) -> Array:
        self.x_shape = x.shape
        self.y_shape = y.shape
        out = xp.maximum(x, y)
        self.out_data = out
        return out

    def backward(self, grad: 'Tensor') -> Tuple[Optional[Array], Optional[Array]]:
        x = self.tensors[0]
        y = self.tensors[1]
        grad_x = None
        grad_y = None
        x_matches = x.data == self.out_data
        y_matches = y.data == self.out_data
        if x.requires_grad:
            grad_x = grad.data * (x_matches * (1.0 - 0.5 * y_matches))
        if y.requires_grad:
            grad_y = grad.data * (y_matches * (1.0 - 0.5 * x_matches))
        if grad_x is not None:
            grad_x = Function.unbroadcast(grad_x, self.x_shape)
        if grad_y is not None:
            grad_y = Function.unbroadcast(grad_y, self.y_shape)
        return (grad_x, grad_y)

class Mean(Function):

    def forward(self, x: Array, axis: Optional[Union[int, Tuple[int, ...]]]=None, keepdims: bool=False) -> Array:
        axis = (axis,) if isinstance(axis, int) else axis
        self.axis = axis
        self.keepdims = keepdims
        return xp.mean(x, axis=axis, keepdims=keepdims)

    def backward(self, grad: 'Tensor') -> Array:
        grad_expanded = grad.expand(self.tensors[0].shape if self.keepdims else self.tensors[0].shape)
        grad_arr = grad_expanded.data
        if self.axis is not None:
            num_elements = xp.prod(xp.array([self.tensors[0].shape[ax] for ax in self.axis]))
        else:
            num_elements = 1
            for dim in self.tensors[0].shape:
                num_elements *= dim
        return grad_arr / num_elements

class Gather(Function):

    def forward(self, x: Array, index: Array) -> Array:
        index_values = xp.asarray(index)
        if index_values.size and (bool(xp.any(index_values < 0)) or bool(xp.any(index_values >= x.shape[0]))):
            raise IndexError('gather index out of bounds')
        out = xp.take(x, index_values, axis=0)
        self.x = x
        self.index = index_values
        return out

    def backward(self, grad: 'Tensor') -> Tuple[Array, None]:
        dx = xp.zeros_like(self.x)
        flat_indices = self.index.reshape(-1)
        flat_grads = grad.data.reshape(-1, dx.shape[1])
        dx = xp.scatter_add(dx, flat_indices, flat_grads)
        return (dx, None)

class View(Function):

    def forward(self, x: Array, new_shape: Union[Tuple[int, ...], List[int]]=(1,)) -> Array:
        if len(new_shape) == 1 and isinstance(new_shape[0], (tuple, list)):
            new_shape = new_shape[0]
        if -1 in new_shape:
            if new_shape.count(-1) > 1:
                raise ValueError('Only one -1 dimension is allowed in shape')
            neg_idx = new_shape.index(-1)
            known_size = xp.prod(xp.array([d for i, d in enumerate(new_shape) if i != neg_idx and d != -1]))
            inferred_size = int(x.size // known_size)
            new_shape = tuple((inferred_size if d == -1 else d for d in new_shape))
        if x.size != xp.prod(xp.array(new_shape)):
            raise ValueError(f'Size of new view must match size of original tensor: {x.size} != {xp.prod(xp.array(new_shape))}')
        self.original_shape = x.shape
        return xp.reshape(x, new_shape)

    def backward(self, grad: Optional['Tensor']) -> Optional[Array]:
        return grad.reshape(*self.original_shape).data if grad is not None else None

class Expand(Function):

    def forward(self, x: Array, shape: Union[Tuple[int, ...], List[int]]=(1,)) -> Array:
        self.original_shape = x.shape
        expanded = xp.broadcast_to(x, shape)
        return expanded

    def backward(self, grad: 'Tensor') -> Array:
        grad_arr = grad.data
        if len(grad_arr.shape) > len(self.original_shape):
            reduce_dims = list(range(len(grad_arr.shape) - len(self.original_shape)))
        else:
            reduce_dims = []
        for i, (self_dim, grad_dim) in enumerate(zip(self.original_shape[::-1], grad_arr.shape[-len(self.original_shape):][::-1])):
            if self_dim == 1 and grad_dim != 1:
                reduce_dims.append(len(grad_arr.shape) - 1 - i)
        if reduce_dims:
            grad_arr = xp.sum(grad_arr, axis=tuple(reduce_dims), keepdims=True)
        if grad_arr.shape != self.original_shape:
            grad_arr = grad_arr.reshape(self.original_shape)
        return grad_arr

class Reshape(Function):

    def forward(self, x: Array, shape: Union[Tuple[int, ...], List[int]]=(1,)) -> Array:
        if len(shape) == 1 and isinstance(shape[0], (tuple, list)):
            shape = shape[0]
        self.original_shape = x.shape
        return xp.reshape(x, shape)

    def backward(self, grad: Optional['Tensor']) -> Optional[Array]:
        return grad.data.reshape(self.original_shape) if grad is not None else None

class Transpose(Function):

    def _get_transpose_axes(self, x: Array, dim0: int, dim1: int) -> Tuple[int, ...]:
        axes = list(range(x.ndim))
        axes[dim0], axes[dim1] = (axes[dim1], axes[dim0])
        return tuple(axes)

    def forward(self, x: Array, dim0: int=0, dim1: int=1) -> Array:
        ndim = x.ndim
        if not (0 <= dim0 < ndim and 0 <= dim1 < ndim):
            raise ValueError(f'Dimensions out of range for tensor with {ndim} dimensions')
        axes = self._get_transpose_axes(x, dim0, dim1)
        self.dim0 = dim0
        self.dim1 = dim1
        return xp.transpose(x, axes)

    def backward(self, grad: 'Tensor') -> Array:
        transposed_grad = xp.transpose(grad.data, self._get_transpose_axes(self.tensors[0].data, self.dim0, self.dim1))
        return transposed_grad

class Pad(Function):

    def forward(self, x: Array, pad_width: Union[int, Tuple[int, int], Tuple[int, int, int, int], Tuple[Tuple[int, int], ...]], mode: str='constant', constant_values: Union[int, float]=0) -> Array:
        normalized_pad_width: list[tuple[int, int]] = []
        if isinstance(pad_width, int):
            normalized_pad_width = [(pad_width, pad_width) for _ in range(x.ndim)]
        elif isinstance(pad_width, (tuple, list)):
            if len(pad_width) == 2 and (not isinstance(pad_width[0], (tuple, list))):
                flat_pad_width = cast(tuple[int, int], tuple(pad_width))
                normalized_pad_width = [(0, 0) for _ in range(x.ndim - 1)] + [(flat_pad_width[0], flat_pad_width[1])]
            elif len(pad_width) == 4 and (not isinstance(pad_width[0], (tuple, list))):
                spatial_pad_width = cast(tuple[int, int, int, int], tuple(pad_width))
                normalized_pad_width = [(0, 0) for _ in range(x.ndim - 2)] + [(spatial_pad_width[2], spatial_pad_width[3]), (spatial_pad_width[0], spatial_pad_width[1])]
            else:
                normalized_pad_width = [(int(p[0]), int(p[1])) for p in cast(Sequence[Sequence[int]], pad_width)]
        else:
            normalized_pad_width = []
        self.pad_width = normalized_pad_width
        self.out_data = xp.pad(x, normalized_pad_width, mode=mode, constant_values=constant_values)
        return self.out_data

    def backward(self, grad: 'Tensor') -> Array:
        slices = tuple((slice(p[0], s - p[1]) for s, p in zip(self.out_data.shape, self.pad_width)))
        return grad.data[slices]

class Cat(Function):

    def forward(self, *tensors: Array, axis: int=0) -> Array:
        self.axis = axis
        self.original_shapes = [t.shape for t in tensors]
        return xp.concatenate(list(tensors), axis=axis)

    def backward(self, grad: 'Tensor') -> Tuple[Optional[Array], ...]:
        grads = []
        start_idx = 0
        for t, shape in zip(self.tensors, self.original_shapes):
            if t.requires_grad:
                slice_idx = [slice(None)] * len(shape)
                slice_idx[self.axis] = slice(start_idx, start_idx + shape[self.axis])
                grad_slice = grad.data[tuple(slice_idx)]
                grads.append(grad_slice)
            else:
                grads.append(None)
            start_idx += shape[self.axis]
        return tuple(grads)

class Permute(Function):

    def forward(self, x: Array, dims: Sequence[int]) -> Array:
        if len(dims) == 1 and isinstance(dims[0], (tuple, list)):
            dims = dims[0]
        self.dims = dims
        return xp.transpose(x, dims)

    def backward(self, grad: 'Tensor') -> Array:
        inv_dims = [self.dims.index(i) for i in range(len(self.dims))]
        return xp.transpose(grad.data, inv_dims)

class Stack(Function):

    def forward(self, *tensors: Array, axis: int=0) -> Array:
        if not tensors:
            raise ValueError('Need at least one tensor to stack')
        expanded_arrays = [xp.expand_dims(t, axis=axis) for t in tensors]
        stacked_data = xp.concatenate(expanded_arrays, axis=axis)
        self.axis = axis
        return stacked_data

    def backward(self, grad: 'Tensor') -> Tuple[Optional[Array], ...]:
        grad_size = grad.shape[self.axis]
        chunk_size = grad_size // len(self.tensors)
        grads = []
        for i, tensor in enumerate(self.tensors):
            if not tensor.requires_grad:
                continue
            idx = [slice(None)] * grad.ndim
            idx[self.axis] = slice(i * chunk_size, (i + 1) * chunk_size)
            grad_slice = grad.data[tuple(idx)]
            grads.append(grad_slice.reshape(tensor.shape))
        return tuple(grads)

class StridedWindows(Function):

    def forward(self, x: Array, kernel_size: int, stride: int) -> Array:
        batch_size, channels, height, width = x.shape
        H_out = (height - kernel_size) // stride + 1
        W_out = (width - kernel_size) // stride + 1
        self.kernel_size = kernel_size
        self.stride = stride
        self.H_out = H_out
        self.W_out = W_out
        self.batch_size = batch_size
        self.channels = channels
        self.data_shape = x.shape
        if hasattr(x, 'strides') and hasattr(x, 'itemsize'):
            base_strides = tuple((int(s // x.itemsize) for s in x.strides))
        else:
            x = xp.array(x)
            base_strides = [1] * x.ndim
            for dim in range(x.ndim - 2, -1, -1):
                base_strides[dim] = base_strides[dim + 1] * x.shape[dim + 1]
            base_strides = tuple(base_strides)
        return xp.as_strided_view(x, shape=(H_out, W_out, batch_size, channels, kernel_size, kernel_size), strides=(base_strides[2] * stride, base_strides[3] * stride, base_strides[0], base_strides[1], base_strides[2], base_strides[3]))

    def backward(self, grad: 'Tensor') -> Array:
        grad_arr = grad.data.reshape(self.H_out, self.W_out, self.batch_size, self.channels, self.kernel_size, self.kernel_size)
        grad_arr = grad_arr.transpose(2, 3, 0, 1, 4, 5)
        grad_padded = xp.zeros(self.data_shape, dtype=grad_arr.dtype)
        for i in range(self.kernel_size):
            for j in range(self.kernel_size):
                grad_padded[:, :, i:i + self.H_out * self.stride:self.stride, j:j + self.W_out * self.stride:self.stride] += grad_arr[:, :, :, :, i, j]
        return grad_padded

class Roll(Function):

    def forward(self, x: Array, shifts: int, dims: Optional[int]=None) -> Array:
        self.shifts = shifts
        self.dims = dims
        self.input_shape = x.shape
        return xp.roll(x, shifts, dims)

    def backward(self, grad: 'Tensor') -> Array:
        grad_arr = grad.data
        if grad_arr.ndim == 0:
            grad_arr = xp.full(self.input_shape, grad_arr)
        return xp.roll(grad_arr, -self.shifts, self.dims)
