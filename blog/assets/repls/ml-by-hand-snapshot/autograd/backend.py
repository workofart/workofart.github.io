from __future__ import annotations
import importlib
import os
from collections.abc import Sequence
from typing import Any, Union
import numpy as _host_np
Array = Any
ScalarLike = Union[int, float, bool]
ArrayLike = Union[Array, Sequence[Any], ScalarLike]

def _discover_backend_name() -> str:
    try:
        mx = importlib.import_module('mlx.core')
    except ModuleNotFoundError:
        mx = None
    if mx is not None:
        _ = mx.default_device()
        return 'mlx'
    try:
        cp = importlib.import_module('cupy')
    except ModuleNotFoundError:
        cp = None
    if cp is not None and cp.cuda.runtime.getDeviceCount() > 0:
        return 'cupy'
    return 'numpy'
_env_backend = os.getenv('AUTOGRAD_BACKEND')
if _env_backend is not None:
    NAME = _env_backend.lower()
    if NAME == 'cupy':
        cp = importlib.import_module('cupy')
        if cp.cuda.runtime.getDeviceCount() <= 0:
            raise RuntimeError('AUTOGRAD_BACKEND=cupy requested, but no CUDA device was detected')
else:
    NAME = _discover_backend_name()
IS_MLX = NAME == 'mlx'
IS_NUMPY = NAME == 'numpy'
IS_CUPY = NAME == 'cupy'
xp: Any
if IS_NUMPY:
    xp = _host_np
    ARRAY_TYPE = xp.ndarray
elif IS_CUPY:
    xp = importlib.import_module('cupy')
    ARRAY_TYPE = xp.ndarray
elif IS_MLX:
    xp = importlib.import_module('mlx.core')
    ARRAY_TYPE = type(xp.array(0, dtype=xp.float32))
else:
    raise ValueError(f'Unknown backend: {NAME}')

def _array(obj: Any, dtype: Any | None=None):
    return xp.asarray(obj, dtype=dtype)

def _scatter_add(dst: Any, idx: Any, updates: Any):
    if IS_MLX:
        return dst.at[idx].add(updates)
    out = xp.array(dst)
    xp.add.at(out, idx, updates)
    return out

def eval(*xs: Any) -> None:
    if IS_MLX and xs:
        xp.eval(*xs)

def _to_scalar(x: Any) -> Any:
    if hasattr(x, 'detach') and hasattr(x, 'cpu'):
        return x.detach().cpu().item()
    eval(x)
    if hasattr(x, 'item'):
        return x.item()
    return x

def _as_strided_view(x, *, shape, strides):
    if IS_MLX:
        return xp.as_strided(x, shape=shape, strides=strides)
    return xp.lib.stride_tricks.as_strided(x, shape=shape, strides=tuple((s * x.itemsize for s in strides)))

def _sample_categorical(logits):
    if logits.ndim != 1:
        raise ValueError('sample_categorical only supports 1-D logits')
    if IS_MLX:
        return xp.random.categorical(logits)
    shifted = logits - xp.max(logits)
    probs = xp.exp(shifted)
    probs = probs / xp.sum(probs)
    return xp.random.choice(probs.shape[-1], p=probs)
_native_random_normal: Any = xp.random.normal
_native_random_bernoulli: Any = getattr(xp.random, 'bernoulli', None)
_native_random_binomial: Any = getattr(xp.random, 'binomial', None)

def _random_normal(loc: float=0.0, scale: float=1.0, size: Any | None=None, *, shape: Any | None=None):
    target_shape = shape if shape is not None else size
    if IS_MLX:
        return _native_random_normal(shape=target_shape, loc=loc, scale=scale)
    return _native_random_normal(loc=loc, scale=scale, size=target_shape)

def _random_bernoulli(p: float, size: Any | None=None, *, shape: Any | None=None):
    target_shape = shape if shape is not None else size
    if IS_MLX:
        return _native_random_bernoulli(p, shape=target_shape)
    return _native_random_binomial(1, p, size=target_shape)

def _to_numpy(x: Any):
    if hasattr(x, 'detach') and hasattr(x, 'cpu'):
        return x.detach().cpu().numpy()
    if IS_MLX:
        eval(x)
        return _host_np.asarray(x)
    if IS_CUPY:
        if hasattr(xp, 'asnumpy'):
            return xp.asnumpy(x)
        if hasattr(x, 'get'):
            return x.get()
    return _host_np.asarray(x)
if not hasattr(xp, 'array'):
    xp.array = _array
if not hasattr(xp, 'scatter_add'):
    xp.scatter_add = _scatter_add
xp.to_scalar = _to_scalar
xp.as_strided_view = _as_strided_view
xp.sample_categorical = _sample_categorical
xp.random.normal = _random_normal
xp.random.bernoulli = _random_bernoulli
xp.to_numpy = _to_numpy
