#!/usr/bin/env python3
"""Recover a sparse matrix from unlabeled two-dimensional Hadamard spectra."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import tempfile
from fractions import Fraction
from pathlib import Path
from typing import Any


SIZE = 8
MAX_SPARSITY = 12
MAX_FRAMES = 3
SPECTRUM_LIMIT = 4096


class ContractError(ValueError):
    """The supplied spectral packet is not a valid recovery case."""


class DuplicateKeyError(ContractError):
    """A JSON object repeats a key."""


def _pairs(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise DuplicateKeyError(key)
        result[key] = value
    return result


def load_payload(path: Path) -> dict[str, Any]:
    try:
        raw = path.read_bytes()
        value = json.loads(
            raw.decode("utf-8"), object_pairs_hook=_pairs
        )
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise ContractError("input") from exc
    if not isinstance(value, dict):
        raise ContractError("object")
    return value


def expect_keys(value: dict[str, Any], expected: set[str]) -> None:
    if set(value) != expected:
        raise ContractError("keys")


def integer(value: Any, low: int, high: int) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise ContractError("integer")
    if value < low or value > high:
        raise ContractError("range")
    return value


def text(value: Any) -> str:
    if not isinstance(value, str) or not value:
        raise ContractError("text")
    return value


def integer_vector(
    value: Any, length: int, low: int, high: int
) -> list[int]:
    if not isinstance(value, list) or len(value) != length:
        raise ContractError("vector")
    return [integer(item, low, high) for item in value]


def spectrum_matrix(value: Any) -> list[list[int]]:
    if not isinstance(value, list) or len(value) != SIZE:
        raise ContractError("spectrum")
    rows = []
    for raw_row in value:
        if not isinstance(raw_row, list) or len(raw_row) != SIZE:
            raise ContractError("spectrum")
        rows.append(
            [
                integer(item, -SPECTRUM_LIMIT, SPECTRUM_LIMIT)
                for item in raw_row
            ]
        )
    return rows


def validate(payload: dict[str, Any]) -> dict[str, Any]:
    expect_keys(
        payload,
        {
            "schema_version",
            "case_id",
            "size",
            "sparsity",
            "coefficient_abs_max",
            "frames",
            "axis_checks",
            "limits",
        },
    )
    integer(payload["schema_version"], 1, 1)
    case_id = text(payload["case_id"])
    size = integer(payload["size"], SIZE, SIZE)
    sparsity = integer(payload["sparsity"], SIZE, MAX_SPARSITY)
    coefficient_bound = integer(payload["coefficient_abs_max"], 1, 20)

    raw_limits = payload["limits"]
    if not isinstance(raw_limits, dict):
        raise ContractError("limits")
    expect_keys(
        raw_limits, {"max_size", "max_sparsity", "max_frames"}
    )
    integer(raw_limits["max_size"], SIZE, SIZE)
    integer(raw_limits["max_sparsity"], MAX_SPARSITY, MAX_SPARSITY)
    integer(raw_limits["max_frames"], MAX_FRAMES, MAX_FRAMES)

    raw_frames = payload["frames"]
    if not isinstance(raw_frames, list) or len(raw_frames) != MAX_FRAMES:
        raise ContractError("frames")
    frames = []
    frame_ids: set[str] = set()
    for raw_frame in raw_frames:
        if not isinstance(raw_frame, dict):
            raise ContractError("frame")
        expect_keys(raw_frame, {"frame_id", "spectrum"})
        frame_id = text(raw_frame["frame_id"])
        if frame_id in frame_ids:
            raise ContractError("frame_id")
        frame_ids.add(frame_id)
        frames.append(
            {
                "frame_id": frame_id,
                "spectrum": spectrum_matrix(raw_frame["spectrum"]),
            }
        )
    frames.sort(key=lambda item: item["frame_id"])

    raw_checks = payload["axis_checks"]
    if not isinstance(raw_checks, dict):
        raise ContractError("axis_checks")
    expect_keys(
        raw_checks,
        {
            "row_energy_moments",
            "column_energy_moments",
            "row_sign_parities",
            "column_sign_parities",
            "global_sign_parity",
        },
    )
    checks = {
        "row_energy_moments": integer_vector(
            raw_checks["row_energy_moments"], SIZE, -(10**18), 10**18
        ),
        "column_energy_moments": integer_vector(
            raw_checks["column_energy_moments"], SIZE, -(10**18), 10**18
        ),
        "row_sign_parities": integer_vector(
            raw_checks["row_sign_parities"], SIZE, 0, 1
        ),
        "column_sign_parities": integer_vector(
            raw_checks["column_sign_parities"], SIZE, 0, 1
        ),
        "global_sign_parity": integer(
            raw_checks["global_sign_parity"], 0, 1
        ),
    }
    return {
        "case_id": case_id,
        "size": size,
        "sparsity": sparsity,
        "coefficient_bound": coefficient_bound,
        "frames": frames,
        "axis_checks": checks,
    }


def solve_exact(
    matrix: list[list[int]], rhs: list[int]
) -> list[Fraction]:
    size = len(rhs)
    augmented = [
        [Fraction(value) for value in matrix[row]]
        + [Fraction(rhs[row])]
        for row in range(size)
    ]
    for column in range(size):
        pivot = next(
            (
                row
                for row in range(column, size)
                if augmented[row][column] != 0
            ),
            None,
        )
        if pivot is None:
            raise ContractError("singular_moments")
        augmented[column], augmented[pivot] = (
            augmented[pivot],
            augmented[column],
        )
        divisor = augmented[column][column]
        for index in range(column, size + 1):
            augmented[column][index] /= divisor
        for row in range(size):
            if row == column:
                continue
            factor = augmented[row][column]
            if factor == 0:
                continue
            for index in range(column, size + 1):
                augmented[row][index] -= factor * augmented[column][index]
    return [augmented[row][size] for row in range(size)]


def energies_from_moments(moments: list[int]) -> list[Fraction]:
    matrix = [
        [pow(row + 1, power) for row in range(SIZE)]
        for power in range(SIZE)
    ]
    return solve_exact(matrix, moments)


def fwht(values: list[int]) -> list[int]:
    result = list(values)
    width = 1
    while width < len(result):
        step = width * 2
        for start in range(0, len(result), step):
            for offset in range(width):
                left = result[start + offset]
                right = result[start + offset + width]
                result[start + offset] = left + right
                result[start + offset + width] = left - right
        width = step
    return result


def inverse_spectrum(spectrum: list[list[int]]) -> list[list[int]]:
    row_transformed = [fwht(row) for row in spectrum]
    column_transformed = [[0] * SIZE for _ in range(SIZE)]
    for column in range(SIZE):
        values = fwht(
            [row_transformed[row][column] for row in range(SIZE)]
        )
        for row, value in enumerate(values):
            column_transformed[row][column] = value
    divisor = SIZE * SIZE
    result = []
    for row in column_transformed:
        output_row = []
        for value in row:
            if value % divisor:
                raise ContractError("nonintegral_inverse")
            output_row.append(value // divisor)
        result.append(output_row)
    return result


def unique_axis_map(
    observed_energies: list[int], canonical_energies: list[Fraction]
) -> list[int]:
    if len(set(observed_energies)) != SIZE:
        raise ContractError("axis_ambiguity")
    mapping = []
    for energy in observed_energies:
        matches = [
            index
            for index, canonical in enumerate(canonical_energies)
            if canonical == energy
        ]
        if len(matches) != 1:
            raise ContractError("axis_moments")
        mapping.append(matches[0])
    if sorted(mapping) != list(range(SIZE)):
        raise ContractError("axis_moments")
    return mapping


def recover_frame_view(
    frame: dict[str, Any],
    row_energies: list[Fraction],
    column_energies: list[Fraction],
) -> dict[str, Any]:
    observed = inverse_spectrum(frame["spectrum"])
    row_map = unique_axis_map(
        [sum(abs(value) for value in row) for row in observed],
        row_energies,
    )
    column_map = unique_axis_map(
        [
            sum(abs(observed[row][column]) for row in range(SIZE))
            for column in range(SIZE)
        ],
        column_energies,
    )
    absolute = [[0] * SIZE for _ in range(SIZE)]
    signs = [[0] * SIZE for _ in range(SIZE)]
    for observed_row, canonical_row in enumerate(row_map):
        for observed_column, canonical_column in enumerate(column_map):
            value = observed[observed_row][observed_column]
            absolute[canonical_row][canonical_column] = abs(value)
            signs[canonical_row][canonical_column] = 1 if value < 0 else 0
    return {
        "frame_id": frame["frame_id"],
        "row_permutation": row_map,
        "column_permutation": column_map,
        "absolute": absolute,
        "signs": signs,
    }


def gf2_unique(
    variable_count: int,
    equations: list[tuple[int, int]],
) -> list[int]:
    basis: dict[int, tuple[int, int]] = {}
    for mask, rhs in equations:
        current_mask = mask
        current_rhs = rhs & 1
        while current_mask:
            pivot = current_mask.bit_length() - 1
            existing = basis.get(pivot)
            if existing is None:
                basis[pivot] = (current_mask, current_rhs)
                break
            current_mask ^= existing[0]
            current_rhs ^= existing[1]
        if current_mask == 0 and current_rhs:
            raise ContractError("sign_conflict")
    if len(basis) != variable_count:
        raise ContractError("sign_ambiguity")
    solution: list[int | None] = [None] * variable_count
    for pivot in sorted(basis):
        mask, rhs = basis[pivot]
        value = rhs
        lower = mask & ((1 << pivot) - 1)
        while lower:
            bit = lower.bit_length() - 1
            solved = solution[bit]
            if solved is None:
                raise ContractError("sign_system")
            value ^= solved
            lower ^= 1 << bit
        solution[pivot] = value
    if any(value is None for value in solution):
        raise ContractError("sign_ambiguity")
    complete = [int(value) for value in solution]
    for mask, rhs in equations:
        parity = 0
        remaining = mask
        while remaining:
            bit = remaining.bit_length() - 1
            parity ^= complete[bit]
            remaining ^= 1 << bit
        if parity != rhs:
            raise ContractError("sign_system")
    return complete


def solve_phase(
    signs: list[list[int]],
    row_parities: list[int],
    column_parities: list[int],
    global_parity: int,
    support: list[tuple[int, int]],
) -> tuple[list[int], list[int], dict[tuple[int, int], int]]:
    x_base = 0
    y_base = SIZE
    c_base = SIZE * 2
    equations: list[tuple[int, int]] = []
    row_edges: list[list[int]] = [[] for _ in range(SIZE)]
    column_edges: list[list[int]] = [[] for _ in range(SIZE)]
    neighbors: list[set[int]] = [set() for _ in range(SIZE * 2)]
    for edge_index, (row, column) in enumerate(support):
        equations.append(
            (
                (1 << (x_base + row))
                | (1 << (y_base + column))
                | (1 << (c_base + edge_index)),
                signs[row][column],
            )
        )
        row_edges[row].append(edge_index)
        column_edges[column].append(edge_index)
        neighbors[row].add(SIZE + column)
        neighbors[SIZE + column].add(row)
    for row in range(SIZE):
        mask = 0
        for edge_index in row_edges[row]:
            mask |= 1 << (c_base + edge_index)
        equations.append((mask, row_parities[row]))
    for column in range(SIZE):
        mask = 0
        for edge_index in column_edges[column]:
            mask |= 1 << (c_base + edge_index)
        equations.append((mask, column_parities[column]))
    cycle_mask = 0
    for edge_index, (row, column) in enumerate(support):
        if (row + 2 * column) % 5 == 0:
            cycle_mask |= 1 << (c_base + edge_index)
    equations.append((cycle_mask, global_parity))
    unseen = set(range(SIZE * 2))
    while unseen:
        start = min(unseen)
        stack = [start]
        component = set()
        unseen.remove(start)
        while stack:
            node = stack.pop()
            component.add(node)
            for neighbor in neighbors[node]:
                if neighbor in unseen:
                    unseen.remove(neighbor)
                    stack.append(neighbor)
        rows = [node for node in component if node < SIZE]
        equations.append((1 << (x_base + min(rows)), 0))
    solution = gf2_unique(SIZE * 2 + len(support), equations)
    row_signs = [
        1 if solution[x_base + row] == 0 else -1
        for row in range(SIZE)
    ]
    column_signs = [
        1 if solution[y_base + column] == 0 else -1
        for column in range(SIZE)
    ]
    coefficient_signs = {
        support[index]: solution[c_base + index]
        for index in range(len(support))
    }
    return row_signs, column_signs, coefficient_signs


def build_result(parsed: dict[str, Any]) -> dict[str, Any]:
    checks = parsed["axis_checks"]
    row_energies = energies_from_moments(
        checks["row_energy_moments"]
    )
    column_energies = energies_from_moments(
        checks["column_energy_moments"]
    )
    for energy in row_energies + column_energies:
        if energy.denominator != 1 or energy <= 0:
            raise ContractError("axis_energy")

    views = [
        recover_frame_view(frame, row_energies, column_energies)
        for frame in parsed["frames"]
    ]
    base_absolute = views[0]["absolute"]
    support = [
        (row, column)
        for row in range(SIZE)
        for column in range(SIZE)
        if base_absolute[row][column]
    ]
    if len(support) != parsed["sparsity"]:
        raise ContractError("support_count")
    if any(
        not any(base_absolute[row][column] for column in range(SIZE))
        for row in range(SIZE)
    ):
        raise ContractError("empty_row")
    if any(
        not any(base_absolute[row][column] for row in range(SIZE))
        for column in range(SIZE)
    ):
        raise ContractError("empty_column")

    coefficient_values: list[int] | None = None
    output_frames = []
    for view in views:
        if view["absolute"] != base_absolute:
            raise ContractError("global_support")
        row_signs, column_signs, coefficient_signs = solve_phase(
            view["signs"],
            checks["row_sign_parities"],
            checks["column_sign_parities"],
            checks["global_sign_parity"],
            support,
        )
        candidate = [
            base_absolute[row][column]
            * (-1 if coefficient_signs[(row, column)] else 1)
            for row, column in support
        ]
        if coefficient_values is None:
            coefficient_values = candidate
        elif coefficient_values != candidate:
            raise ContractError("frame_disagreement")
        for (row, column), value in zip(support, candidate):
            expected_sign = (
                row_signs[row]
                * column_signs[column]
                * (1 if value > 0 else -1)
            )
            actual_sign = -1 if view["signs"][row][column] else 1
            if expected_sign != actual_sign:
                raise ContractError("phase")
        output_frames.append(
            {
                "frame_id": view["frame_id"],
                "row_permutation": view["row_permutation"],
                "column_permutation": view["column_permutation"],
                "row_signs": row_signs,
                "column_signs": column_signs,
            }
        )

    if coefficient_values is None:
        raise ContractError("no_frames")
    if any(
        abs(value) > parsed["coefficient_bound"]
        for value in coefficient_values
    ):
        raise ContractError("coefficient_bound")
    matrix = [
        {"row": row, "column": column, "value": value}
        for (row, column), value in zip(support, coefficient_values)
    ]
    return {
        "schema_version": 1,
        "case_id": parsed["case_id"],
        "matrix": matrix,
        "frames": output_frames,
        "support_score": sum(abs(value) for value in coefficient_values),
    }


def canonical_bytes(value: dict[str, Any]) -> bytes:
    text_value = json.dumps(
        value, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    )
    return (text_value + "\n").encode("utf-8")


def read_optional(path: Path) -> bytes | None:
    if not path.exists():
        return None
    if not path.is_file():
        raise ContractError("path")
    return path.read_bytes()


def stage(path: Path, data: bytes, mode: int) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    temporary_path = Path(temporary)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(data)
            handle.flush()
            os.fsync(handle.fileno())
        os.chmod(temporary_path, mode)
    except BaseException:
        if temporary_path.exists():
            temporary_path.unlink()
        raise
    return temporary_path


def restore(path: Path, old: bytes | None) -> None:
    if old is None:
        try:
            path.unlink()
        except FileNotFoundError:
            pass
        return
    temporary = stage(path, old, 0o644)
    try:
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def ledger_prefix(path: Path, line: bytes) -> bytes:
    existing = read_optional(path)
    if existing is None:
        return line
    if line in existing.splitlines(keepends=True):
        return existing
    if existing and not existing.endswith(b"\n"):
        raise ContractError("ledger")
    return existing + line


def commit(
    output_path: Path,
    output_bytes: bytes,
    ledger_path: Path,
    planned_ledger: bytes,
) -> None:
    old_output = read_optional(output_path)
    old_ledger = read_optional(ledger_path)
    output_temporary = stage(output_path, output_bytes, 0o644)
    ledger_temporary: Path | None = None
    output_replaced = False
    try:
        if planned_ledger != (old_ledger or b""):
            ledger_temporary = stage(ledger_path, planned_ledger, 0o600)
        os.replace(output_temporary, output_path)
        output_replaced = True
        if ledger_temporary is not None:
            os.replace(ledger_temporary, ledger_path)
            ledger_temporary = None
    except BaseException:
        if output_replaced:
            restore(output_path, old_output)
        raise
    finally:
        if output_temporary.exists():
            output_temporary.unlink()
        if ledger_temporary is not None and ledger_temporary.exists():
            ledger_temporary.unlink()


def execute(args: argparse.Namespace) -> None:
    input_path = Path(args.input)
    output_path = Path(args.output)
    ledger_path = Path(args.ledger)
    raw_input = input_path.read_bytes()
    parsed = validate(load_payload(input_path))
    result = build_result(parsed)
    result_bytes = canonical_bytes(result)
    event = canonical_bytes(
        {
            "schema_version": 1,
            "event": "completed",
            "case_id": parsed["case_id"],
            "input_sha256": hashlib.sha256(raw_input).hexdigest(),
            "result_sha256": hashlib.sha256(result_bytes).hexdigest(),
        }
    )
    planned_ledger = ledger_prefix(ledger_path, event)
    commit(output_path, result_bytes, ledger_path, planned_ledger)


def parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--ledger", required=True)
    return parser.parse_args(argv)


def main() -> int:
    try:
        execute(parse_args(sys.argv[1:]))
    except Exception:
        return 3
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
