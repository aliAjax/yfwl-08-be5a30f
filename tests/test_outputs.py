from __future__ import annotations

import copy
import hashlib
import json
import os
import subprocess
from pathlib import Path

import pytest


ENTRY = os.environ.get("YEL008_ENTRY", "/app/walsh_recover.py")
SIZE = 8
MATRIX = [
    (0, 6, 3),
    (0, 7, -6),
    (1, 6, -1),
    (2, 1, -7),
    (2, 6, -6),
    (3, 0, -12),
    (4, 2, 2),
    (4, 3, -16),
    (5, 4, 19),
    (6, 5, 3),
    (7, 2, 13),
    (7, 3, 9),
]
FRAMES = [
    {
        "frame_id": "F1",
        "row_permutation": [2, 7, 0, 5, 3, 1, 6, 4],
        "column_permutation": [5, 0, 7, 2, 1, 6, 4, 3],
        "row_signs": [1, -1, 1, 1, -1, 1, -1, 1],
        "column_signs": [1, 1, -1, -1, 1, -1, 1, 1],
    },
    {
        "frame_id": "F2",
        "row_permutation": [4, 1, 6, 0, 7, 3, 5, 2],
        "column_permutation": [2, 6, 0, 5, 3, 7, 1, 4],
        "row_signs": [1, 1, -1, 1, 1, -1, 1, -1],
        "column_signs": [-1, 1, 1, -1, 1, 1, -1, 1],
    },
    {
        "frame_id": "F3",
        "row_permutation": [6, 3, 1, 7, 2, 5, 0, 4],
        "column_permutation": [7, 3, 6, 1, 4, 0, 2, 5],
        "row_signs": [1, 1, 1, -1, -1, 1, -1, 1],
        "column_signs": [1, -1, 1, 1, -1, 1, 1, -1],
    },
]


def _matrix(cells: list[tuple[int, int, int]] = MATRIX) -> list[list[int]]:
    result = [[0] * SIZE for _ in range(SIZE)]
    for row, column, value in cells:
        result[row][column] = value
    return result


def _hadamard() -> list[list[int]]:
    result = [[1]]
    while len(result) < SIZE:
        result = [
            row + row for row in result
        ] + [
            row + [-value for value in row] for row in result
        ]
    return result


def _multiply(
    left: list[list[int]], right: list[list[int]]
) -> list[list[int]]:
    return [
        [
            sum(
                left[row][inner] * right[inner][column]
                for inner in range(SIZE)
            )
            for column in range(SIZE)
        ]
        for row in range(SIZE)
    ]


def _spectrum(
    frame: dict, cells: list[tuple[int, int, int]] = MATRIX
) -> list[list[int]]:
    base = _matrix(cells)
    signed = [[0] * SIZE for _ in range(SIZE)]
    for observed_row, canonical_row in enumerate(
        frame["row_permutation"]
    ):
        for observed_column, canonical_column in enumerate(
            frame["column_permutation"]
        ):
            signed[observed_row][observed_column] = (
                frame["row_signs"][canonical_row]
                * frame["column_signs"][canonical_column]
                * base[canonical_row][canonical_column]
            )
    transform = _hadamard()
    return _multiply(_multiply(transform, signed), transform)


def _energy_moments(energies: list[int]) -> list[int]:
    return [
        sum(pow(index + 1, power) * energies[index] for index in range(SIZE))
        for power in range(SIZE)
    ]


def _axis_checks(
    cells: list[tuple[int, int, int]] = MATRIX,
) -> dict:
    base = _matrix(cells)
    row_energies = [
        sum(abs(value) for value in row) for row in base
    ]
    column_energies = [
        sum(abs(base[row][column]) for row in range(SIZE))
        for column in range(SIZE)
    ]
    return {
        "row_energy_moments": _energy_moments(row_energies),
        "column_energy_moments": _energy_moments(column_energies),
        "row_sign_parities": [
            sum(1 for value in row if value < 0) % 2 for row in base
        ],
        "column_sign_parities": [
            sum(1 for row in base if row[column] < 0) % 2
            for column in range(SIZE)
        ],
        "global_sign_parity": (
            sum(
                value < 0
                for row, column, value in cells
                if (row + 2 * column) % 5 == 0
            )
            % 2
        ),
    }


def _expected_signs() -> list[tuple[list[int], list[int]]]:
    neighbors = [set() for _ in range(SIZE * 2)]
    for row, column, value in MATRIX:
        neighbors[row].add(SIZE + column)
        neighbors[SIZE + column].add(row)
    unseen = set(range(SIZE * 2))
    result = []
    for frame in FRAMES:
        row_signs = list(frame["row_signs"])
        column_signs = list(frame["column_signs"])
        remaining = set(unseen)
        while remaining:
            start = min(remaining)
            stack = [start]
            component = set()
            remaining.remove(start)
            while stack:
                node = stack.pop()
                component.add(node)
                for neighbor in neighbors[node]:
                    if neighbor in remaining:
                        remaining.remove(neighbor)
                        stack.append(neighbor)
            rows = [node for node in component if node < SIZE]
            columns = [node - SIZE for node in component if node >= SIZE]
            minimum = min(rows)
            if row_signs[minimum] < 0:
                row_signs = [
                    -value if index in rows else value
                    for index, value in enumerate(row_signs)
                ]
                column_signs = [
                    -value if index in columns else value
                    for index, value in enumerate(column_signs)
                ]
        result.append((row_signs, column_signs))
    return result


def _payload(
    case_id: str = "spectra-0815",
    cells: list[tuple[int, int, int]] = MATRIX,
) -> dict:
    return {
        "schema_version": 1,
        "case_id": case_id,
        "size": SIZE,
        "sparsity": len(cells),
        "coefficient_abs_max": 20,
        "frames": [
            {
                "frame_id": frame["frame_id"],
                "spectrum": _spectrum(frame, cells),
            }
            for frame in FRAMES
        ],
        "axis_checks": _axis_checks(cells),
        "limits": {
            "max_size": SIZE,
            "max_sparsity": 12,
            "max_frames": 3,
        },
    }


def _invoke(
    directory: Path,
    payload: dict,
    entry: str = ENTRY,
    existing_output: bytes | None = None,
    existing_ledger: bytes | None = None,
) -> tuple[subprocess.CompletedProcess[str], Path, Path, Path]:
    directory.mkdir(parents=True, exist_ok=True)
    input_path = directory / "spectra.json"
    output_path = directory / "recovered.json"
    ledger_path = directory / "spectra-ledger.jsonl"
    input_path.write_text(json.dumps(payload), encoding="utf-8")
    if existing_output is not None:
        output_path.write_bytes(existing_output)
    if existing_ledger is not None:
        ledger_path.write_bytes(existing_ledger)
    completed = subprocess.run(
        [
            "python3",
            entry,
            "--input",
            str(input_path),
            "--output",
            str(output_path),
            "--ledger",
            str(ledger_path),
        ],
        capture_output=True,
        text=True,
        timeout=60,
        env=os.environ.copy(),
    )
    return completed, output_path, ledger_path, input_path


def _raw_invoke(
    directory: Path, raw: bytes, entry: str = ENTRY
) -> tuple[subprocess.CompletedProcess[str], Path, Path]:
    directory.mkdir(parents=True, exist_ok=True)
    input_path = directory / "raw-spectra.json"
    output_path = directory / "raw-recovered.json"
    ledger_path = directory / "raw-ledger.jsonl"
    input_path.write_bytes(raw)
    completed = subprocess.run(
        [
            "python3",
            entry,
            "--input",
            str(input_path),
            "--output",
            str(output_path),
            "--ledger",
            str(ledger_path),
        ],
        capture_output=True,
        text=True,
        timeout=60,
        env=os.environ.copy(),
    )
    return completed, output_path, ledger_path


@pytest.fixture(scope="session")
def solved_case(tmp_path_factory: pytest.TempPathFactory) -> dict:
    directory = tmp_path_factory.mktemp("solved-walsh-case")
    completed, output_path, ledger_path, input_path = _invoke(
        directory, _payload()
    )
    raw = output_path.read_bytes()
    return {
        "directory": directory,
        "completed": completed,
        "result": json.loads(raw),
        "raw": raw,
        "ledger_path": ledger_path,
        "input_path": input_path,
        "payload": _payload(),
    }


def test_recovers_global_sparse_matrix(solved_case: dict):
    """题面：「`matrix` 恰好含 `sparsity` 项，按 `row`、`column` 递增。」"""
    expected = [
        {"row": row, "column": column, "value": value}
        for row, column, value in MATRIX
    ]
    assert solved_case["completed"].returncode == 0
    assert solved_case["result"]["matrix"] == expected


def test_recovers_first_frame_permutations(solved_case: dict):
    """题面：「`row_permutation[o]` 是观测行 `o` 对应的规范行」。"""
    frame = solved_case["result"]["frames"][0]
    assert solved_case["completed"].returncode == 0
    assert frame["row_permutation"] == FRAMES[0]["row_permutation"]
    assert frame["column_permutation"] == FRAMES[0]["column_permutation"]


def test_recovers_second_frame_permutations(solved_case: dict):
    """题面：「`column_permutation[o]` 是观测列 `o` 对应的规范列」。"""
    frame = solved_case["result"]["frames"][1]
    assert solved_case["completed"].returncode == 0
    assert frame["row_permutation"] == FRAMES[1]["row_permutation"]
    assert frame["column_permutation"] == FRAMES[1]["column_permutation"]


def test_recovers_third_frame_permutations(solved_case: dict):
    """题面：「三张频谱的置换和符号独立，底片矩阵 `C` 相同。」"""
    frame = solved_case["result"]["frames"][2]
    assert solved_case["completed"].returncode == 0
    assert frame["row_permutation"] == FRAMES[2]["row_permutation"]
    assert frame["column_permutation"] == FRAMES[2]["column_permutation"]


def test_recovers_canonical_row_signs(solved_case: dict):
    """题面：「每个分量内最小行号的 `row_signs` 必须为 `1`」。"""
    actual = [frame["row_signs"] for frame in solved_case["result"]["frames"]]
    expected = [signs[0] for signs in _expected_signs()]
    assert solved_case["completed"].returncode == 0
    assert actual == expected


def test_recovers_canonical_column_signs(solved_case: dict):
    """题面：「符号数组每项为 `1` 或 `-1`。」"""
    actual = [
        frame["column_signs"] for frame in solved_case["result"]["frames"]
    ]
    expected = [signs[1] for signs in _expected_signs()]
    assert solved_case["completed"].returncode == 0
    assert actual == expected


def test_reports_global_support_score(solved_case: dict):
    """题面：「`support_score` 等于全部 `abs(value)` 之和。」"""
    assert solved_case["completed"].returncode == 0
    assert solved_case["result"]["support_score"] == 97


def test_accepts_full_twelve_sparsity_limit(tmp_path: Path):
    """题面：「`sparsity` 最高可取 `12`；读到 `13` 或更大值时不再继续，过程以状态码 `3` 结束。」"""
    completed, output_path, _, _ = _invoke(tmp_path, _payload())
    result = json.loads(output_path.read_text(encoding="utf-8"))
    assert completed.returncode == 0
    assert len(result["matrix"]) == 12
    assert result["support_score"] == 97
    assert result["matrix"][0] == {"row": 0, "column": 6, "value": 3}


def test_rejects_inconsistent_energy_moments(tmp_path: Path):
    """题面：「能量矩与谱面矛盾时，同样返回 `3`。」"""
    payload = copy.deepcopy(_payload())
    payload["axis_checks"]["row_energy_moments"][0] += 1
    completed, output_path, ledger_path, _ = _invoke(tmp_path, payload)
    assert completed.returncode == 3
    assert not output_path.exists()
    assert not ledger_path.exists()


def test_rejects_inconsistent_sign_parities(tmp_path: Path):
    """题面：「`row_sign_parities[i]` 是第 `i` 行所有非零系数符号位的异或」。"""
    payload = copy.deepcopy(_payload())
    payload["axis_checks"]["row_sign_parities"][0] ^= 1
    completed, output_path, ledger_path, _ = _invoke(tmp_path, payload)
    assert completed.returncode == 3
    assert not output_path.exists()
    assert not ledger_path.exists()


def test_rejects_nonintegral_inverse_spectrum(tmp_path: Path):
    """题面：「逆变换若不能还原成整数底片，复核到此为止，状态码为 `3`。」"""
    payload = copy.deepcopy(_payload())
    payload["frames"][0]["spectrum"][0][0] += 1
    completed, output_path, ledger_path, _ = _invoke(tmp_path, payload)
    assert completed.returncode == 3
    assert not output_path.exists()
    assert not ledger_path.exists()


def test_rejects_global_support_disagreement(tmp_path: Path):
    """题面：「三帧的绝对值支撑不能逐格重合时，同样按 `3` 退回。」"""
    payload = copy.deepcopy(_payload())
    payload["frames"][0]["spectrum"] = [
        [value + 1 for value in row]
        for row in payload["frames"][0]["spectrum"]
    ]
    completed, output_path, ledger_path, _ = _invoke(tmp_path, payload)
    assert completed.returncode == 3
    assert not output_path.exists()
    assert not ledger_path.exists()


def test_rejects_top_level_extra_key(tmp_path: Path):
    """题面：「输入的根对象只允许：」"""
    payload = copy.deepcopy(_payload())
    payload["comment"] = "extra"
    completed, output_path, ledger_path, _ = _invoke(tmp_path, payload)
    assert completed.returncode == 3
    assert not output_path.exists()
    assert not ledger_path.exists()


def test_rejects_missing_limits_key(tmp_path: Path):
    """题面：「`limits` 的字段固定为：」"""
    payload = copy.deepcopy(_payload())
    payload.pop("limits")
    completed, output_path, ledger_path, _ = _invoke(tmp_path, payload)
    assert completed.returncode == 3
    assert not output_path.exists()
    assert not ledger_path.exists()


def test_wrong_schema_version_is_rejected(tmp_path: Path):
    """题面：「`schema_version` 必须是整数 `1`。」"""
    payload = copy.deepcopy(_payload())
    payload["schema_version"] = 2
    completed, output_path, ledger_path, _ = _invoke(tmp_path, payload)
    assert completed.returncode == 3
    assert not output_path.exists()
    assert not ledger_path.exists()


def test_rejects_extra_frame_key(tmp_path: Path):
    """题面：「`frames` 恰好含三张频谱。」"""
    payload = copy.deepcopy(_payload())
    payload["frames"][0]["quality"] = 1
    completed, output_path, ledger_path, _ = _invoke(tmp_path, payload)
    assert completed.returncode == 3
    assert not output_path.exists()
    assert not ledger_path.exists()


def test_rejects_duplicate_frame_id(tmp_path: Path):
    """题面：「`frame_id` 非空且三张互异。」"""
    payload = copy.deepcopy(_payload())
    payload["frames"][1]["frame_id"] = payload["frames"][0]["frame_id"]
    completed, output_path, ledger_path, _ = _invoke(tmp_path, payload)
    assert completed.returncode == 3
    assert not output_path.exists()
    assert not ledger_path.exists()


def test_rejects_boolean_spectrum_value(tmp_path: Path):
    """题面：「每帧的 `spectrum` 是 8×8 频点表：外层有 8 条记录，每条记录含 8 个频点读数」。"""
    payload = copy.deepcopy(_payload())
    payload["frames"][0]["spectrum"][0][0] = True
    completed, output_path, ledger_path, _ = _invoke(tmp_path, payload)
    assert completed.returncode == 3
    assert not output_path.exists()
    assert not ledger_path.exists()


def test_rejects_spectrum_above_limit(tmp_path: Path):
    """题面：「单元格落在闭区间 `[-4096, 4096]` 内。」"""
    payload = copy.deepcopy(_payload())
    payload["frames"][0]["spectrum"][0][0] = 4097
    completed, output_path, ledger_path, _ = _invoke(tmp_path, payload)
    assert completed.returncode == 3
    assert not output_path.exists()
    assert not ledger_path.exists()


def test_rejects_wrong_size(tmp_path: Path):
    """题面：「`size` 固定为 `8`。`sparsity` 是 `8` 至 `12` 的整数。」"""
    payload = copy.deepcopy(_payload())
    payload["size"] = 4
    completed, output_path, ledger_path, _ = _invoke(tmp_path, payload)
    assert completed.returncode == 3
    assert not output_path.exists()
    assert not ledger_path.exists()


def test_rejects_sparsity_thirteen(tmp_path: Path):
    """题面：「`sparsity` 最高可取 `12`；读到 `13` 或更大值时不再继续，过程以状态码 `3` 结束。」"""
    payload = copy.deepcopy(_payload())
    payload["sparsity"] = 13
    completed, output_path, ledger_path, _ = _invoke(tmp_path, payload)
    assert completed.returncode == 3
    assert not output_path.exists()
    assert not ledger_path.exists()


def test_rejects_fourth_frame(tmp_path: Path):
    """题面：「`frames` 恰好含三张频谱。」"""
    payload = copy.deepcopy(_payload())
    payload["frames"].append(copy.deepcopy(payload["frames"][0]))
    completed, output_path, ledger_path, _ = _invoke(tmp_path, payload)
    assert completed.returncode == 3
    assert not output_path.exists()
    assert not ledger_path.exists()


def test_rejects_extra_axis_check_key(tmp_path: Path):
    """题面：「`axis_checks` 字段固定为：」"""
    payload = copy.deepcopy(_payload())
    payload["axis_checks"]["global_hash"] = 1
    completed, output_path, ledger_path, _ = _invoke(tmp_path, payload)
    assert completed.returncode == 3
    assert not output_path.exists()
    assert not ledger_path.exists()


def test_rejects_nonbinary_sign_parity(tmp_path: Path):
    """题面：「每个奇偶值只能是整数 `0` 或 `1`。」"""
    payload = copy.deepcopy(_payload())
    payload["axis_checks"]["column_sign_parities"][0] = 2
    completed, output_path, ledger_path, _ = _invoke(tmp_path, payload)
    assert completed.returncode == 3
    assert not output_path.exists()
    assert not ledger_path.exists()


def test_rejects_missing_frame(tmp_path: Path):
    """题面：「三张频谱必须对应同一个 8×8 稀疏整数系数矩阵。」"""
    payload = copy.deepcopy(_payload())
    payload["frames"].pop()
    completed, output_path, ledger_path, _ = _invoke(tmp_path, payload)
    assert completed.returncode == 3
    assert not output_path.exists()
    assert not ledger_path.exists()


def test_duplicate_json_key_is_rejected(tmp_path: Path):
    """题面：「JSON 不允许重复键覆盖旧值；出现重复键同样退出 `3`。」"""
    raw = b'{"schema_version":1,"schema_version":1,"case_id":"x"}'
    completed, output_path, ledger_path = _raw_invoke(tmp_path, raw)
    assert completed.returncode == 3
    assert not output_path.exists()
    assert not ledger_path.exists()


def test_emits_exact_result_schema(solved_case: dict):
    """题面：「成功结果写到 `--output`。根字段固定为：」"""
    result = solved_case["result"]
    assert solved_case["completed"].returncode == 0
    assert set(result) == {
        "schema_version",
        "case_id",
        "matrix",
        "frames",
        "support_score",
    }
    assert set(result["matrix"][0]) == {"row", "column", "value"}
    assert set(result["frames"][0]) == {
        "frame_id",
        "row_permutation",
        "column_permutation",
        "row_signs",
        "column_signs",
    }


def test_result_is_compact_sorted_json(tmp_path: Path):
    """题面：「`--output` 写成单行 UTF-8 JSON：对象键按字典序排列，分隔符旁不添空格，文件以唯一换行收尾。」"""
    completed, output_path, _, _ = _invoke(tmp_path, _payload())
    raw = output_path.read_bytes()
    assert completed.returncode == 0
    assert b", " not in raw
    assert raw.endswith(b"\n")
    assert not raw.endswith(b"\n\n")
    assert list(json.loads(raw)) == sorted(json.loads(raw))


def test_ledger_has_exact_keys_and_result_digest(solved_case: dict):
    """题面：「原始票据和复核文件各生成一条小写 SHA-256 摘要。」"""
    record = json.loads(
        solved_case["ledger_path"].read_text(encoding="utf-8")
    )
    assert solved_case["completed"].returncode == 0
    assert set(record) == {
        "case_id",
        "event",
        "input_sha256",
        "result_sha256",
        "schema_version",
    }
    assert record["result_sha256"] == hashlib.sha256(
        solved_case["raw"]
    ).hexdigest()


def test_repeating_identical_ticket_does_not_append(solved_case: dict):
    """题面：「同一票据重复复核只会复用既有账本内容，旧账本字节保持不变。」"""
    before = solved_case["ledger_path"].read_bytes()
    completed, _, _, _ = _invoke(
        solved_case["directory"], solved_case["payload"]
    )
    assert completed.returncode == 0
    assert solved_case["ledger_path"].read_bytes() == before


def test_new_ticket_appends_record(tmp_path: Path):
    """题面：「新的成功批次要追加一行，每写完一条事件立即追加单个换行。」"""
    first, _, ledger_path, _ = _invoke(tmp_path, _payload("spectra-a"))
    first_count = len(ledger_path.read_text(encoding="utf-8").splitlines())
    second, _, _, _ = _invoke(tmp_path, _payload("spectra-b"))
    second_count = len(ledger_path.read_text(encoding="utf-8").splitlines())
    assert first.returncode == 0
    assert second.returncode == 0
    assert first_count == 1
    assert second_count == 2


def test_invalid_ticket_preserves_previous_bytes(tmp_path: Path):
    """题面：「被退回的调用不向控制台写字符，`--output` 与 `--ledger` 都维持调用前的字节。」"""
    payload = copy.deepcopy(_payload())
    payload["frames"].pop()
    completed, output_path, ledger_path, _ = _invoke(
        tmp_path,
        payload,
        existing_output=b"old-result\n",
        existing_ledger=b"old-ledger\n",
    )
    assert completed.returncode == 3
    assert output_path.read_bytes() == b"old-result\n"
    assert ledger_path.read_bytes() == b"old-ledger\n"


def test_install_failure_preserves_directory_and_ledger(tmp_path: Path):
    """题面：「替换若失败，旧结果和旧账本仍可原样读取，旁路文件也不会残留。」"""
    payload = _payload()
    input_path = tmp_path / "spectra.json"
    output_path = tmp_path / "recovered.json"
    ledger_path = tmp_path / "ledger.jsonl"
    input_path.write_text(json.dumps(payload), encoding="utf-8")
    output_path.mkdir()
    completed = subprocess.run(
        [
            "python3",
            ENTRY,
            "--input",
            str(input_path),
            "--output",
            str(output_path),
            "--ledger",
            str(ledger_path),
        ],
        capture_output=True,
        text=True,
        timeout=60,
        env=os.environ.copy(),
    )
    assert completed.returncode == 3
    assert output_path.is_dir()
    assert not ledger_path.exists()


def test_malformed_ledger_preserves_output(tmp_path: Path):
    """题面：「旧账本字节保持不变。」"""
    completed, output_path, ledger_path, _ = _invoke(
        tmp_path,
        _payload(),
        existing_output=b"prior\n",
        existing_ledger=b"{broken",
    )
    assert completed.returncode == 3
    assert output_path.read_bytes() == b"prior\n"
    assert ledger_path.read_bytes() == b"{broken"


def test_success_path_is_silent(solved_case: dict):
    """题面：「成功复核不向标准输出写字符，标准错误也保持空白。」"""
    assert solved_case["completed"].returncode == 0
    assert solved_case["completed"].stdout == ""
    assert solved_case["completed"].stderr == ""


def test_case_identifier_is_preserved(solved_case: dict):
    """题面：「`case_id` 是非空字符串。」"""
    assert solved_case["completed"].returncode == 0
    assert solved_case["result"]["case_id"] == "spectra-0815"


def test_rejects_axis_energy_tie(tmp_path: Path):
    """题面：「行、列能量若打平，规范轴不再唯一，仍以 `3` 退出。」"""
    cells = [
        (0, 0, 2),
        (1, 1, 2),
        (2, 2, 3),
        (3, 3, 4),
        (4, 4, 5),
        (5, 5, 6),
        (6, 6, 7),
        (7, 7, 8),
    ]
    completed, output_path, ledger_path, _ = _invoke(
        tmp_path, _payload(cells=cells)
    )
    assert completed.returncode == 3
    assert not output_path.exists()
    assert not ledger_path.exists()
