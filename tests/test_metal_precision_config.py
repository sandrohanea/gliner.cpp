"""Check the Metal precision overlay without requiring a Metal device."""

import subprocess
import sys
import tempfile
from pathlib import Path


ORIGINAL = "    return !ggml_is_transposed(op->src[0]) &&"
FIXED = ("    return !(op->src[0]->type == GGML_TYPE_F32 &&\n"
         "             op->src[1]->type == GGML_TYPE_F32 &&\n"
         "             ggml_get_op_params_i32(op, 0) == GGML_PREC_F32) &&\n"
         "           !ggml_is_transposed(op->src[0]) &&")


def main(cmake, module):
    with tempfile.TemporaryDirectory() as temp:
        root = Path(temp)
        source = root / "ggml-metal-common.cpp"
        original = "bool use_mm() {\n" + ORIGINAL + "\n        true;\n}\n"
        source.write_text(original)
        script = root / "transform.cmake"
        script.write_text(f'include("{Path(module).resolve().as_posix()}")\n'
                          'gliner_metal_f32_source("${INPUT}" "${OUTPUT}")\n')
        output = root / "patched.cpp"

        def transform(ok):
            result = subprocess.run([cmake, f"-DINPUT={source}", f"-DOUTPUT={output}",
                                     "-P", str(script)], capture_output=True, text=True)
            assert (result.returncode == 0) == ok, result.stdout + result.stderr
            return result

        transform(True)
        assert source.read_text() == original
        assert output.read_text() == original.replace(ORIGINAL, FIXED)
        patched = output.read_bytes()
        transform(True)
        assert output.read_bytes() == patched
        source.write_bytes(patched)
        transform(True)
        assert output.read_bytes() == patched
        source.write_text("bool unrelated() { return true; }\n")
        assert "Unsupported GGML Metal matrix dispatch" in transform(False).stderr
        source.write_text(original + original)
        assert "Unsupported GGML Metal matrix dispatch" in transform(False).stderr
        source.write_text(original + patched.decode())
        assert "Mixed patched and unpatched" in transform(False).stderr
    print("Metal precision overlay is repeatable and leaves source checkouts unchanged")


if __name__ == "__main__":
    main(*sys.argv[1:])
