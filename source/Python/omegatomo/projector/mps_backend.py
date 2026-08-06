# -*- coding: utf-8 -*-
"""Native PyTorch-MPS bridge for OMEGA projectorType123 Metal kernels.

This first implementation intentionally targets the SPECT configuration used by
SPECT_main_DIP_PyTorch_MPS.py:

* projector types 1--3 (projectorType123)
* float32 buffers, no integer accumulator conversion
* listmode disabled
* useImages disabled
* no attenuation, normalization, scatter/additional corrections, masks or TOF
* one reconstructed volume

The forward and backward Metal pipelines are compiled once during projector
initialization because the same entry point is specialized with different FP/BP
macros and the active OMEGA user settings.
"""

from __future__ import annotations

import hashlib
import os
import re
import shlex
from pathlib import Path
import struct
from typing import Any, Iterable

import numpy as np


# Metal/C++ SIMD layout for kernelParams.hpp::ScalarKernelParams.
# float3/int3/uint3 occupy and align to 16 bytes in MSL and simd:: types.
SCALAR_KERNEL_PARAMS_SIZE = 336

_OFFSETS = {
    "nRowsD": 0,
    "nColsD": 4,
    "dPitch": 8,
    "dL": 16,
    "global_factor": 20,
    "epps": 24,
    "det_per_ring": 28,
    "sigma_x": 32,
    "coneOfResponseStdCoeffA": 36,
    "coneOfResponseStdCoeffB": 40,
    "coneOfResponseStdCoeffC": 44,
    "tube_width": 48,
    "cylRadiusProj3": 52,
    "bmin": 56,
    "bmax": 60,
    "Vmax": 64,
    "rings": 68,
    "helicalRadius": 72,
    "totalFOVmin": 80,
    "totalFOVmax": 96,
    "d_N": 112,
    "b": 128,
    "dSize5": 144,
    "kerroin4": 152,
    "DSC": 156,
    "d": 160,
    "d_Scale4": 176,
    "d_Scale5": 192,
    "d_bmax": 208,
    "orthWidth": 224,
    "nProjections": 232,
    "no_norm": 240,
    "m_size": 248,
    "currentSubset": 256,
    "aa": 260,
    "N_PDHG": 272,
    "epps_PDHG": 288,
    "theta_PDHG": 292,
    "tau_PDHG": 296,
    "enforcePositivity_PDHG": 300,
    "N_rotate": 304,
    "cosa_rotate": 320,
    "sina_rotate": 324,
}


def _scalar(value: Any, default: float = 0.0) -> float:
    if value is None:
        return float(default)
    arr = np.asarray(value)
    if arr.size == 0:
        return float(default)
    return float(arr.reshape(-1)[0])


def _indexed_scalar(obj: Any, name: str, index: int, default: float = 0.0) -> float:
    value = getattr(obj, name, None)
    if value is None:
        return float(default)
    arr = np.asarray(value)
    if arr.size == 0:
        return float(default)
    flat = arr.reshape(-1)
    if flat.size == 1:
        return float(flat[0])
    if index >= flat.size:
        return float(default)
    return float(flat[index])


def _int_scalar(value: Any, default: int = 0) -> int:
    if value is None:
        return int(default)
    arr = np.asarray(value)
    if arr.size == 0:
        return int(default)
    return int(arr.reshape(-1)[0])


def _indexed_int(obj: Any, name: str, index: int, default: int = 0) -> int:
    value = getattr(obj, name, None)
    if value is None:
        return int(default)
    arr = np.asarray(value)
    if arr.size == 0:
        return int(default)
    flat = arr.reshape(-1)
    if flat.size == 1:
        return int(flat[0])
    if index >= flat.size:
        return int(default)
    return int(flat[index])


def _pack_scalar_kernel_params(self: Any, subset: int, volume: int) -> bytes:
    """Pack ScalarKernelParams using the ABI in kernelParams.hpp.

    Add a host-side static_assert(sizeof(ScalarKernelParams) == 336) to the
    existing Metal backend when integrating this patch permanently.
    """

    blob = bytearray(SCALAR_KERNEL_PARAMS_SIZE)

    def u32(name: str, value: int) -> None:
        struct.pack_into("<I", blob, _OFFSETS[name], int(value))

    def i32(name: str, value: int) -> None:
        struct.pack_into("<i", blob, _OFFSETS[name], int(value))

    def i64(name: str, value: int) -> None:
        struct.pack_into("<q", blob, _OFFSETS[name], int(value))

    def u64(name: str, value: int) -> None:
        struct.pack_into("<Q", blob, _OFFSETS[name], int(value))

    def u8(name: str, value: int) -> None:
        struct.pack_into("<B", blob, _OFFSETS[name], int(value))

    def f32(name: str, value: float) -> None:
        struct.pack_into("<f", blob, _OFFSETS[name], float(value))

    def f2(name: str, values: Iterable[float]) -> None:
        a, b = values
        struct.pack_into("<2f", blob, _OFFSETS[name], float(a), float(b))

    def f3(name: str, values: Iterable[float]) -> None:
        a, b, c = values
        # The fourth float is ABI padding for Metal simd::float3/float3.
        struct.pack_into("<4f", blob, _OFFSETS[name], float(a), float(b), float(c), 0.0)

    def u3(name: str, values: Iterable[int]) -> None:
        a, b, c = values
        struct.pack_into("<4I", blob, _OFFSETS[name], int(a), int(b), int(c), 0)

    def i3(name: str, values: Iterable[int]) -> None:
        a, b, c = values
        struct.pack_into("<4i", blob, _OFFSETS[name], int(a), int(b), int(c), 0)

    nx = _indexed_int(self, "Nx", volume, 1)
    ny = _indexed_int(self, "Ny", volume, 1)
    nz = _indexed_int(self, "Nz", volume, 1)
    dx = _indexed_scalar(self, "dx", volume, 1.0)
    dy = _indexed_scalar(self, "dy", volume, 1.0)
    dz = _indexed_scalar(self, "dz", volume, 1.0)
    bx = _indexed_scalar(self, "bx", volume, 0.0)
    by = _indexed_scalar(self, "by", volume, 0.0)
    bz = _indexed_scalar(self, "bz", volume, 0.0)

    u32("nRowsD", _int_scalar(self.nRowsD))
    u32("nColsD", _int_scalar(self.nColsD))
    f2("dPitch", (_scalar(self.dPitchX), _scalar(self.dPitchY)))
    f32("dL", _scalar(getattr(self, "dL", 0.0)))
    f32("global_factor", _scalar(getattr(self, "global_factor", 1.0), 1.0))
    f32("epps", _scalar(getattr(self, "epps", 1e-5), 1e-5))
    u32("det_per_ring", _int_scalar(getattr(self, "det_per_ring", 0)))
    f32("sigma_x", _scalar(getattr(self, "sigma_x", 0.0)))
    f32("coneOfResponseStdCoeffA", _scalar(getattr(self, "coneOfResponseStdCoeffA", 0.0)))
    f32("coneOfResponseStdCoeffB", _scalar(getattr(self, "coneOfResponseStdCoeffB", 0.0)))
    f32("coneOfResponseStdCoeffC", _scalar(getattr(self, "coneOfResponseStdCoeffC", 0.0)))
    f32("tube_width", _scalar(getattr(self, "tube_width_z", 0.0)))
    f32("cylRadiusProj3", _scalar(getattr(self, "tube_radius", 0.0)))
    f32("bmin", _scalar(getattr(self, "bmin", 0.0)))
    f32("bmax", _scalar(getattr(self, "bmax", 0.0)))
    f32("Vmax", _scalar(getattr(self, "Vmax", 0.0)))
    u32("rings", _int_scalar(getattr(self, "rings", 0)))
    f32("helicalRadius", _scalar(getattr(self, "helicalRadius", 0.0)))
    f3(
        "totalFOVmin",
        (
            _scalar(getattr(self, "totalFOVxmin", 0.0)),
            _scalar(getattr(self, "totalFOVymin", 0.0)),
            _scalar(getattr(self, "totalFOVzmin", 0.0)),
        ),
    )
    f3(
        "totalFOVmax",
        (
            _scalar(getattr(self, "totalFOVxmax", 0.0)),
            _scalar(getattr(self, "totalFOVymax", 0.0)),
            _scalar(getattr(self, "totalFOVzmax", 0.0)),
        ),
    )

    u3("d_N", (nx, ny, nz))
    f3("b", (bx, by, bz))
    f2(
        "dSize5",
        (
            _indexed_scalar(self, "dSizeX", volume, 0.0),
            _indexed_scalar(self, "dSizeY", volume, 0.0),
        ),
    )
    f32("kerroin4", _indexed_scalar(self, "kerroin", volume, 0.0))
    f32("DSC", 0.0)
    f3("d", (dx, dy, dz))
    f3(
        "d_Scale4",
        (
            _indexed_scalar(self, "dScaleX4", volume, 0.0),
            _indexed_scalar(self, "dScaleY4", volume, 0.0),
            _indexed_scalar(self, "dScaleZ4", volume, 0.0),
        ),
    )
    f3(
        "d_Scale5",
        (
            _indexed_scalar(self, "dScaleX", volume, 0.0),
            _indexed_scalar(self, "dScaleY", volume, 0.0),
            _indexed_scalar(self, "dScaleZ", volume, 0.0),
        ),
    )
    f3("d_bmax", (bx + nx * dx, by + ny * dy, bz + nz * dz))
    f32("orthWidth", _scalar(getattr(self, "tube_width_z", 0.0)))
    i64("nProjections", _indexed_int(self, "nProjSubset", subset, 0))
    u8("no_norm", _int_scalar(getattr(self, "no_norm", 1), 1))
    u64("m_size", _indexed_int(self, "nMeasSubset", subset, 0))
    u32("currentSubset", subset)
    i32("aa", volume)

    i3("N_PDHG", (0, 0, 0))
    f32("epps_PDHG", 0.0)
    f32("theta_PDHG", 0.0)
    f32("tau_PDHG", 0.0)
    u8("enforcePositivity_PDHG", 0)
    i3("N_rotate", (0, 0, 0))
    f32("cosa_rotate", 0.0)
    f32("sina_rotate", 0.0)

    return bytes(blob)


def _mps_tensor_from_numpy(torch: Any, value: Any, dtype: Any) -> Any:
    arr = np.asarray(value, dtype=dtype)
    return torch.as_tensor(np.ascontiguousarray(arr), device="mps")


def _validate_configuration(self: Any) -> None:
    unsupported = []
    if not getattr(self, "SPECT", False):
        unsupported.append("only SPECT is enabled in this first bridge")
    if getattr(self, "FPType", None) not in (1, 2, 3):
        unsupported.append("forward projector must be type 1, 2 or 3")
    if getattr(self, "BPType", None) not in (1, 2, 3):
        unsupported.append("backprojector must be type 1, 2 or 3")
    if getattr(self, "useImages", False):
        unsupported.append("useImages must be False")
    if getattr(self, "listmode", 0) != 0:
        unsupported.append("listmode must be disabled")
    if getattr(self, "TOF", False):
        unsupported.append("TOF is not yet wired")
    if getattr(self, "attenuation_correction", False):
        unsupported.append("attenuation correction is not yet wired")
    if getattr(self, "normalization_correction", False):
        unsupported.append("normalization correction is not yet wired")
    if getattr(self, "additionalCorrection", False):
        unsupported.append("additional/scatter correction is not yet wired")
    if getattr(self, "useMaskFP", False) or getattr(self, "useMaskBP", False):
        unsupported.append("forward/backprojection masks are not yet wired")
    if getattr(self, "use_psf", False):
        unsupported.append("separate PSF convolution is not yet wired")
    if getattr(self, "nMultiVolumes", 0) != 0:
        unsupported.append("multi-volume reconstruction is not yet wired")
    if getattr(self, "use_64bit_atomics", False) or getattr(self, "use_32bit_atomics", False):
        unsupported.append("integer accumulator conversion is not yet wired")
    if unsupported:
        raise NotImplementedError("Metal/MPS bridge configuration: " + "; ".join(unsupported))




_SHADER_CACHE: dict[str, Any] = {}
_LOCAL_INCLUDE_RE = re.compile(r'^\s*#\s*include\s*"([^"]+)"\s*(?://.*)?$')


def _iter_option_tokens(options: Iterable[Any]) -> Iterable[str]:
    """Yield command-line-like tokens from OMEGA's bOpt tuples."""
    for option in options:
        text = str(option).strip()
        if not text:
            continue
        try:
            tokens = shlex.split(text)
        except ValueError:
            tokens = text.split()
        for token in tokens:
            yield token.strip()


def _macro_preamble(options: Iterable[Any]) -> str:
    """Convert OMEGA ``-D`` options into Metal source definitions.

    Metal uses the ScalarKernelParams ABI directly.  The CUDA-only ``PYTHON``
    macro must not be active, because it enables the legacy individual-scalar
    reconstruction block in projectorType123.cl and duplicates variables that
    are already produced by UNPACK_SCALAR_PARAMS_123.
    """
    definitions: dict[str, str | None] = {}
    order: list[str] = []

    for token in _iter_option_tokens(options):
        if not token.startswith('-D') or len(token) <= 2:
            continue
        body = token[2:]
        if '=' in body:
            name, value = body.split('=', 1)
        else:
            name, value = body, None
        name = name.strip()
        if not name or name == 'PYTHON':
            continue
        if name not in definitions:
            order.append(name)
        definitions[name] = value.strip() if value is not None else None

    if 'METAL' not in definitions:
        order.insert(0, 'METAL')
        definitions['METAL'] = None

    lines = ['// Generated from OMEGA projector settings.']
    for name in order:
        value = definitions[name]
        if value is None or value == '':
            lines.append(f'#define {name} 1')
        else:
            lines.append(f'#define {name} {value}')
    return '\n'.join(lines) + '\n\n'


def _resolve_local_include(name: str, current_dir: Path, search_dirs: tuple[Path, ...]) -> Path:
    candidates = [current_dir / name]
    candidates.extend(directory / name for directory in search_dirs)
    for candidate in candidates:
        candidate = candidate.resolve()
        if candidate.is_file():
            return candidate
    searched = ', '.join(str(path) for path in candidates)
    raise FileNotFoundError(f'Unable to resolve Metal include {name!r}; searched: {searched}')


def _inline_local_includes(
    source: str,
    *,
    current_dir: Path,
    search_dirs: tuple[Path, ...],
    active_stack: tuple[Path, ...] = (),
) -> str:
    """Inline quoted local includes while preserving Metal/system includes.

    ``kernelParams.hpp`` is injected explicitly before the common OMEGA source,
    so any source-level include of that file is skipped here.  This guarantees
    that ScalarKernelParams is declared before SCALAR_PARAMS is used and avoids
    duplicate struct definitions.
    """
    output: list[str] = []
    for line in source.splitlines():
        match = _LOCAL_INCLUDE_RE.match(line)
        if match is None:
            output.append(line)
            continue

        include_name = match.group(1)
        if Path(include_name).name == 'kernelParams.hpp':
            output.append('// kernelParams.hpp injected by mps_backend.py')
            continue

        include_path = _resolve_local_include(include_name, current_dir, search_dirs)
        if include_path in active_stack:
            chain = ' -> '.join(str(path) for path in (*active_stack, include_path))
            raise RuntimeError(f'Recursive Metal include detected: {chain}')

        include_text = include_path.read_text(encoding='utf-8')
        output.append(f'// BEGIN INLINED INCLUDE: {include_path}')
        output.append(
            _inline_local_includes(
                include_text,
                current_dir=include_path.parent,
                search_dirs=search_dirs,
                active_stack=(*active_stack, include_path),
            )
        )
        output.append(f'// END INLINED INCLUDE: {include_path}')
    return '\n'.join(output)


def _find_kernel_params(root: Path, search_dirs: tuple[Path, ...]) -> Path:
    candidates = [root / 'kernelParams.hpp']
    candidates.extend(directory / 'kernelParams.hpp' for directory in search_dirs)
    candidates.append(Path(__file__).resolve().with_name('kernelParams.hpp'))
    for candidate in candidates:
        candidate = candidate.resolve()
        if candidate.is_file():
            return candidate
    searched = ', '.join(str(path) for path in candidates)
    raise FileNotFoundError(f'kernelParams.hpp was not found; searched: {searched}')


def _metal_address_space_fixes(source: str) -> str:
    """Apply Metal-only pointer address-space corrections for type-123 SPECT.

    getDetectorCoordinatesSPECT accepts the geometry arrays as ``device``
    pointers.  The generic kernel declaration used ``CONSTANT`` for d_xy/d_z,
    which expands to Metal's ``constant`` address space and cannot be passed to
    that helper.  These two read-only arrays are ordinary MPS buffers, so
    ``const device`` is the correct declaration.
    """
    patterns = {
        r'\bCONSTANT\s+float\s*\*\s*d_xy\s*\[\[buffer\(8\)\]\]':
            'const CLGLOBAL float* d_xy [[buffer(8)]]',
        r'\bCONSTANT\s+float\s*\*\s*d_z\s*\[\[buffer\(9\)\]\]':
            'const CLGLOBAL float* d_z [[buffer(9)]]',
    }
    for pattern, replacement in patterns.items():
        source, count = re.subn(pattern, replacement, source)
        if count == 0:
            raise RuntimeError(
                f'Unable to locate required Metal kernel declaration matching {pattern!r}'
            )
    return source


def _assemble_metal_source(
    source_body: str,
    compiler_options: Iterable[Any],
    source_root: os.PathLike[str] | str,
) -> str:
    root = Path(source_root).resolve()
    if not root.is_dir():
        raise FileNotFoundError(f'Metal source directory does not exist: {root}')

    search_dirs = (root, root.parent, root / 'include')
    expanded = _inline_local_includes(
        source_body,
        current_dir=root,
        search_dirs=search_dirs,
    )
    expanded = _metal_address_space_fixes(expanded)

    kernel_params_path = root.parent / 'cpp/kernelParams.hpp' #_find_kernel_params(root, search_dirs)
    kernel_params = kernel_params_path.read_text(encoding='utf-8')
    # Wrap older unguarded copies, but do not nest the same guard around the
    # guarded drop-in header because that would suppress the struct body.
    if 'OMEGA_KERNEL_PARAMS_HPP_INCLUDED' not in kernel_params:
        kernel_params = (
            '#ifndef OMEGA_KERNEL_PARAMS_HPP_INCLUDED\n'
            '#define OMEGA_KERNEL_PARAMS_HPP_INCLUDED 1\n\n'
            + kernel_params
            + '\n#endif // OMEGA_KERNEL_PARAMS_HPP_INCLUDED\n'
        )

    return (
        _macro_preamble(compiler_options)
        + '#include <metal_stdlib>\nusing namespace metal;\n\n'
        + kernel_params
        + '\n'
        + expanded
        + '\n'
    )


def _compile_shader_cached(torch: Any, source: str, label: str) -> Any:
    digest = hashlib.sha256(source.encode('utf-8')).hexdigest()
    library = _SHADER_CACHE.get(digest)
    if library is None:
        try:
            library = torch.mps.compile_shader(source)
        except Exception as exc:
            raise RuntimeError(
                f'Metal {label} projector compilation failed. '
                f'Source SHA-256: {digest}'
            ) from exc
        _SHADER_CACHE[digest] = library
    return library


def init_mps_projector(
    self: Any,
    *,
    source_root: os.PathLike[str] | str,
    source_fp: str,
    source_bp: str,
    options_fp: Iterable[Any],
    options_bp: Iterable[Any],
) -> None:
    """Compile the Metal FP/BP variants once and move static data to MPS."""
    import torch

    _validate_configuration(self)

    if not torch.backends.mps.is_available():
        raise RuntimeError('PyTorch MPS is not available on this machine.')
    if not hasattr(torch.mps, 'compile_shader'):
        raise RuntimeError(
            'This backend requires a PyTorch version that provides '
            'torch.mps.compile_shader().'
        )

    self.no_norm = 1
    self.mSize = self.nRowsD * self.nColsD * self.nProjections

    complete_fp = _assemble_metal_source(source_fp, options_fp, source_root)
    complete_bp = _assemble_metal_source(source_bp, options_bp, source_root)
    self.mps_lib_fp = _compile_shader_cached(torch, complete_fp, 'forward')
    self.mps_lib_bp = _compile_shader_cached(torch, complete_bp, 'backward')

    try:
        self.knlF = self.mps_lib_fp.projectorType123
        self.knlB = self.mps_lib_bp.projectorType123
    except AttributeError as exc:
        raise RuntimeError(
            "Compiled Metal library does not expose 'projectorType123'."
        ) from exc

    # A reusable placeholder occupies compile-time inactive/gapped Metal slots.
    self.mps_dummy_buffer = torch.zeros(1, dtype=torch.uint8, device='mps')
    self.d_Sens = torch.zeros(1, dtype=torch.float32, device='mps')

    self.d_x = [None] * self.subsets
    self.d_z = [None] * self.subsets
    x_flat = np.asarray(self.x, dtype=np.float32).ravel()
    z_flat = np.asarray(self.z, dtype=np.float32).ravel()
    z_stride = 6 if self.pitch else 2
    for subset in range(self.subsets):
        start = _indexed_int(self, 'nMeas', subset)
        stop = _indexed_int(self, 'nMeas', subset + 1)
        self.d_x[subset] = _mps_tensor_from_numpy(
            torch, x_flat[start * 6 : stop * 6], np.float32
        )
        self.d_z[subset] = _mps_tensor_from_numpy(
            torch, z_flat[start * z_stride : stop * z_stride], np.float32
        )

    self.d_rayShiftsDetector = _mps_tensor_from_numpy(
        torch, self.rayShiftsDetector, np.float32
    )
    self.d_rayShiftsSource = _mps_tensor_from_numpy(
        torch, self.rayShiftsSource, np.float32
    )

    # One immutable ScalarKernelParams buffer per subset/volume.
    self.mps_scalar_params = []
    for subset in range(self.subsets):
        per_volume = []
        for volume in range(self.nMultiVolumes + 1):
            blob = _pack_scalar_kernel_params(self, subset, volume)
            cpu_bytes = np.frombuffer(blob, dtype=np.uint8).copy()
            per_volume.append(torch.as_tensor(cpu_bytes, device='mps'))
        self.mps_scalar_params.append(per_volume)


def _kernel_args(self: Any, scalar_params: Any, dynamic_input: Any, output: Any, subset: int) -> list[Any]:
    """Construct positional arguments matching explicit Metal buffer indices 0..20."""
    args = [self.mps_dummy_buffer] * 21
    args[0] = scalar_params
    args[1] = self.d_rayShiftsDetector
    args[2] = self.d_rayShiftsSource
    args[8] = self.d_x[subset]
    args[9] = self.d_z[subset]
    args[12] = self.d_Sens
    args[19] = dynamic_input
    args[20] = output
    return args


def _require_mps_float32_contiguous(tensor: Any, name: str) -> Any:
    import torch

    if not isinstance(tensor, torch.Tensor):
        raise TypeError(f'{name} must be a PyTorch tensor')
    if tensor.device.type != 'mps':
        raise ValueError(f"{name} must be on device='mps', got {tensor.device}")
    if tensor.dtype != torch.float32:
        raise TypeError(f'{name} must use torch.float32, got {tensor.dtype}')
    return tensor.contiguous()


def forward_projection_mps(self: Any, f: Any, subset: int = -1) -> Any:
    import torch

    if subset == -1:
        subset = self.subset
    subset = int(subset)
    f = _require_mps_float32_contiguous(f, 'forward image')
    if f.numel() != _indexed_int(self, 'N', 0):
        raise ValueError(
            f"Forward image has {f.numel()} elements, expected {_indexed_int(self, 'N', 0)}"
        )

    if self.subsetType > 7 or self.subsets == 1:
        output_size = int(self.nRowsD * self.nColsD * self.nProjSubset[subset].item())
    else:
        output_size = int(self.nMeasSubset[subset].item())
    output = torch.zeros(output_size, dtype=torch.float32, device='mps')

    args = _kernel_args(self, self.mps_scalar_params[subset][0], f, output, subset)
    self.knlF(
        *args,
        threads=tuple(int(v) for v in self.globalSizeFP[subset]),
        group_size=tuple(int(v) for v in self.localSizeFP),
    )
    return output


def backward_projection_mps(self: Any, y: Any, subset: int = -1) -> Any:
    import torch

    if subset == -1:
        subset = self.subset
    subset = int(subset)
    y = _require_mps_float32_contiguous(y, 'backprojection input')

    if self.subsetType > 7 or self.subsets == 1:
        expected = int(self.nRowsD * self.nColsD * self.nProjSubset[subset].item())
    else:
        expected = int(self.nMeasSubset[subset].item())
    if y.numel() != expected:
        raise ValueError(f'Backprojection input has {y.numel()} elements, expected {expected}')

    output = torch.zeros(_indexed_int(self, 'N', 0), dtype=torch.float32, device='mps')
    args = _kernel_args(self, self.mps_scalar_params[subset][0], y, output, subset)
    self.knlB(
        *args,
        threads=tuple(int(v) for v in self.globalSizeBP[subset][0]),
        group_size=tuple(int(v) for v in self.localSizeBP),
    )
    return output
