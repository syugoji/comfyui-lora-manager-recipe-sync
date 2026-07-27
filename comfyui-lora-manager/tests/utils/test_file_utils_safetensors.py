import json
import struct

from py.utils.file_utils import is_valid_safetensors_file


def test_validates_safetensors_container_without_loading_tensors(tmp_path):
    path = tmp_path / "valid.safetensors"
    header = json.dumps({
        "weight": {"dtype": "U8", "shape": [1], "data_offsets": [0, 1]},
        "__metadata__": {"name": "test"},
    }, separators=(",", ":")).encode("utf-8")
    path.write_bytes(struct.pack("<Q", len(header)) + header + b"\x00")

    assert is_valid_safetensors_file(str(path)) is True


def test_rejects_impossible_safetensors_header_length(tmp_path):
    path = tmp_path / "corrupt.safetensors"
    path.write_bytes(struct.pack("<Q", 2**40) + b"not-a-header")

    assert is_valid_safetensors_file(str(path)) is False
