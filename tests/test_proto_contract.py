"""The execution boundary contract compiles and stays decimal-safe.

The contract is defined in protobuf now, while the only executor is in-process
Python, so the C++ engine can implement the same service later with no
renegotiation. A contract that has never been compiled is a contract that does
not exist.
"""

from __future__ import annotations

import re
from importlib.resources import files
from pathlib import Path

import pytest

PROTO = Path(__file__).resolve().parent.parent / "proto" / "execution.proto"


def test_proto_file_exists() -> None:
    assert PROTO.is_file()


def test_proto_compiles(tmp_path: Path) -> None:
    grpc_tools = pytest.importorskip("grpc_tools", reason="grpcio-tools not installed")
    from grpc_tools import protoc

    well_known = str(files(grpc_tools) / "_proto")
    rc = protoc.main(
        [
            "protoc",
            f"-I{PROTO.parent}",
            f"-I{well_known}",
            f"--python_out={tmp_path}",
            f"--grpc_python_out={tmp_path}",
            str(PROTO),
        ]
    )
    assert rc == 0, "proto/execution.proto failed to compile"
    assert (tmp_path / "execution_pb2.py").is_file()
    assert (tmp_path / "execution_pb2_grpc.py").is_file()


def test_no_floating_point_fields_in_the_contract() -> None:
    # "All monetary and ratio values are decimal strings. Never float."
    # a double on the wire would reintroduce binary rounding at the one place
    # two languages have to agree exactly.
    source = PROTO.read_text()
    offenders = [
        f"{lineno}: {line.strip()}"
        for lineno, line in enumerate(source.splitlines(), start=1)
        if re.match(r"\s*(repeated\s+)?(double|float)\s+\w+\s*=", line)
    ]
    assert not offenders, "floating-point fields in the execution contract:\n" + "\n".join(
        offenders
    )


def test_every_referenced_message_is_defined() -> None:
    source = PROTO.read_text()
    defined = set(re.findall(r"^\s*(?:message|enum)\s+(\w+)", source, flags=re.MULTILINE))
    referenced = set(re.findall(r"^\s*(?:repeated\s+)?([A-Z]\w+)\s+\w+\s*=", source, flags=re.MULTILINE))
    rpc_types = set(re.findall(r"rpc\s+\w+\s*\(\s*(?:stream\s+)?(\w+)\s*\)", source))
    rpc_types |= set(re.findall(r"returns\s*\(\s*(?:stream\s+)?(\w+)\s*\)", source))

    missing = (referenced | rpc_types) - defined - {"Timestamp"}
    assert not missing, f"referenced but undefined in execution.proto: {sorted(missing)}"
