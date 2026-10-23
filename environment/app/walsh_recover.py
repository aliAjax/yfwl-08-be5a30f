#!/usr/bin/env python3
"""Legacy spectral catalogue reporter."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import tempfile
from pathlib import Path
from typing import Any


SIZE = 8
MAX_SPARSITY = 12


class InputProblem(ValueError):
    """The spectral packet cannot be interpreted."""


class DuplicateKey(InputProblem):
    """A JSON object repeats a key."""


def pairs_hook(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise DuplicateKey(key)
        result[key] = value
    return result


def read_payload(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(
            path.read_bytes().decode("utf-8"),
            object_pairs_hook=pairs_hook,
        )
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise InputProblem("input") from exc
    if not isinstance(value, dict):
        raise InputProblem("object")
    return value


def exact_keys(value: dict[str, Any], expected: set[str]) -> None:
    if set(value) != expected:
        raise InputProblem("keys")


def whole(value: Any, minimum: int, maximum: int) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise InputProblem("integer")
    if value < minimum or value > maximum:
        raise InputProblem("range")
    return value


def label(value: Any) -> str:
    if not isinstance(value, str) or not value:
        raise InputProblem("label")
    return value


def integer_list(
    value: Any, length: int, minimum: int, maximum: int
) -> list[int]:
    if not isinstance(value, list) or len(value) != length:
        raise InputProblem("list")
    return [whole(item, minimum, maximum) for item in value]


def read_spectrum(value: Any) -> list[list[int]]:
    if not isinstance(value, list) or len(value) != SIZE:
        raise InputProblem("spectrum")
    result = []
    for raw_row in value:
        if not isinstance(raw_row, list) or len(raw_row) != SIZE:
            raise InputProblem("spectrum")
        result.append(integer_list(raw_row, SIZE, -4096, 4096))
    return result


def validate(payload: dict[str, Any]) -> dict[str, Any]:
    exact_keys(
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
    whole(payload["schema_version"], 1, 1)
    case_id = label(payload["case_id"])
    whole(payload["size"], SIZE, SIZE)
    sparsity = whole(payload["sparsity"], SIZE, MAX_SPARSITY)
    coefficient_bound = whole(payload["coefficient_abs_max"], 1, 20)

    raw_limits = payload["limits"]
    if not isinstance(raw_limits, dict):
        raise InputProblem("limits")
    exact_keys(
        raw_limits, {"max_size", "max_sparsity", "max_frames"}
    )
    whole(raw_limits["max_size"], SIZE, SIZE)
    whole(raw_limits["max_sparsity"], MAX_SPARSITY, MAX_SPARSITY)
    whole(raw_limits["max_frames"], 3, 3)

    raw_frames = payload["frames"]
    if not isinstance(raw_frames, list) or len(raw_frames) != 3:
        raise InputProblem("frames")
    frames = []
    frame_ids = set()
    for raw_frame in raw_frames:
        if not isinstance(raw_frame, dict):
            raise InputProblem("frame")
        exact_keys(raw_frame, {"frame_id", "spectrum"})
        frame_id = label(raw_frame["frame_id"])
        if frame_id in frame_ids:
            raise InputProblem("frame_id")
        frame_ids.add(frame_id)
        frames.append(
            {
                "frame_id": frame_id,
                "spectrum": read_spectrum(raw_frame["spectrum"]),
            }
        )
    frames.sort(key=lambda item: item["frame_id"])

    raw_checks = payload["axis_checks"]
    if not isinstance(raw_checks, dict):
        raise InputProblem("axis_checks")
    exact_keys(
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
        "row_energy_moments": integer_list(
            raw_checks["row_energy_moments"],
            SIZE,
            -(10**18),
            10**18,
        ),
        "column_energy_moments": integer_list(
            raw_checks["column_energy_moments"],
            SIZE,
            -(10**18),
            10**18,
        ),
        "row_sign_parities": integer_list(
            raw_checks["row_sign_parities"], SIZE, 0, 1
        ),
        "column_sign_parities": integer_list(
            raw_checks["column_sign_parities"], SIZE, 0, 1
        ),
        "global_sign_parity": whole(
            raw_checks["global_sign_parity"], 0, 1
        ),
    }
    return {
        "case_id": case_id,
        "sparsity": sparsity,
        "coefficient_bound": coefficient_bound,
        "frames": frames,
        "axis_checks": checks,
    }


def catalogue_result(parsed: dict[str, Any]) -> dict[str, Any]:
    positions = [
        (index // SIZE, index % SIZE)
        for index in range(parsed["sparsity"])
    ]
    return {
        "schema_version": 1,
        "case_id": parsed["case_id"],
        "matrix": [
            {"row": row, "column": column, "value": 1}
            for row, column in positions
        ],
        "frames": [
            {
                "frame_id": frame["frame_id"],
                "row_permutation": list(range(SIZE)),
                "column_permutation": list(range(SIZE)),
                "row_signs": [1] * SIZE,
                "column_signs": [1] * SIZE,
            }
            for frame in parsed["frames"]
        ],
        "support_score": parsed["sparsity"],
    }


def compact(value: dict[str, Any]) -> bytes:
    value_text = json.dumps(
        value, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    )
    return (value_text + "\n").encode("utf-8")


def optional_bytes(path: Path) -> bytes | None:
    if not path.exists():
        return None
    if not path.is_file():
        raise InputProblem("path")
    return path.read_bytes()


def staged(path: Path, data: bytes, mode: int) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    temporary = Path(name)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(data)
            handle.flush()
            os.fsync(handle.fileno())
        os.chmod(temporary, mode)
    except BaseException:
        if temporary.exists():
            temporary.unlink()
        raise
    return temporary


def ledger_bytes(path: Path, event: bytes) -> bytes:
    existing = optional_bytes(path)
    if existing is None:
        return event
    if event in existing.splitlines(keepends=True):
        return existing
    if existing and not existing.endswith(b"\n"):
        raise InputProblem("ledger")
    return existing + event


def publish(
    output_path: Path,
    output: bytes,
    ledger_path: Path,
    ledger: bytes,
) -> None:
    old_output = optional_bytes(output_path)
    old_ledger = optional_bytes(ledger_path)
    output_temp = staged(output_path, output, 0o644)
    ledger_temp = None
    try:
        if ledger != (old_ledger or b""):
            ledger_temp = staged(ledger_path, ledger, 0o600)
        os.replace(output_temp, output_path)
        if ledger_temp is not None:
            os.replace(ledger_temp, ledger_path)
            ledger_temp = None
    except BaseException:
        if old_output is None:
            try:
                output_path.unlink()
            except FileNotFoundError:
                pass
        else:
            restore = staged(output_path, old_output, 0o644)
            try:
                os.replace(restore, output_path)
            finally:
                if restore.exists():
                    restore.unlink()
        raise
    finally:
        if output_temp.exists():
            output_temp.unlink()
        if ledger_temp is not None and ledger_temp.exists():
            ledger_temp.unlink()


def execute(args: argparse.Namespace) -> None:
    input_path = Path(args.input)
    output_path = Path(args.output)
    ledger_path = Path(args.ledger)
    raw_input = input_path.read_bytes()
    parsed = validate(read_payload(input_path))
    result = catalogue_result(parsed)
    result_bytes = compact(result)
    event = compact(
        {
            "schema_version": 1,
            "event": "completed",
            "case_id": parsed["case_id"],
            "input_sha256": hashlib.sha256(raw_input).hexdigest(),
            "result_sha256": hashlib.sha256(result_bytes).hexdigest(),
        }
    )
    planned = ledger_bytes(ledger_path, event)
    publish(output_path, result_bytes, ledger_path, planned)


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
