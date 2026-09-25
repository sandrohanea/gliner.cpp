"""Exercise the CUDA source overlay without a CUDA toolkit or GPU."""

import subprocess
import sys
import tempfile
from pathlib import Path


ORIGINAL = "if (ggml_cuda_should_use_mmf(src0->type, cc, warp_size, src0->ne, src0->nb, ne11, /*mul_mat_id =*/ false)) {"
FIXED = ("if (ggml_get_op_params_i32(dst, 3) != GGML_PREC_F32 &&\n        "
         "ggml_cuda_should_use_mmf(src0->type, cc, warp_size, src0->ne, src0->nb, ne11, /*mul_mat_id =*/ false)) {")


def main(cmake, module):
    with tempfile.TemporaryDirectory() as temp:
        root = Path(temp)
        source = root / "ggml-cuda.cu"
        other = root / "other.cu"
        other.write_text("other source\n")
        # Brackets and semicolons must survive CMake's source-list handling.
        original = "void dispatch() {\n    " + ORIGINAL + "\n        multiply();\n    }\n}\n"
        source.write_text(original)
        (root / "CMakeLists.txt").write_text(f"""
cmake_minimum_required(VERSION 3.20)
project(precision_overlay LANGUAGES CXX)
include("{Path(module).resolve().as_posix()}")
add_library(ggml-cuda STATIC ggml-cuda.cu other.cu)
set_target_properties(ggml-cuda PROPERTIES LINKER_LANGUAGE CXX)
gliner_fix_cuda_precision()
get_target_property(sources ggml-cuda SOURCES)
file(WRITE "${{CMAKE_BINARY_DIR}}/sources.txt" "${{sources}}")
get_target_property(includes ggml-cuda INCLUDE_DIRECTORIES)
file(WRITE "${{CMAKE_BINARY_DIR}}/includes.txt" "${{includes}}")
set_source_files_properties(${{sources}} PROPERTIES LANGUAGE CXX)
# Only configure/generate; no CUDA compiler is needed.
""", encoding="utf8")
        # Script mode tests source transformation and its error paths.
        script = root / "transform.cmake"
        script.write_text(
            f'include("{Path(module).resolve().as_posix()}")\n'
            'gliner_cuda_f32_source("${INPUT}" "${OUTPUT}")\n', encoding="utf8")
        output = root / "patched.cu"

        def transform(ok):
            result = subprocess.run([cmake, f"-DINPUT={source.as_posix()}",
                                     f"-DOUTPUT={output.as_posix()}", "-P", str(script)],
                                    capture_output=True, text=True)
            assert (result.returncode == 0) == ok, result.stdout + result.stderr
            return result

        transform(True)
        assert source.read_text() == original, "Modified the user's GGML checkout"
        assert output.read_text() == original.replace(ORIGINAL, FIXED)
        first = output.read_bytes()
        transform(True)
        assert output.read_bytes() == first, "Overlay is not repeatable"
        source.write_bytes(first)
        transform(True)
        assert output.read_bytes() == first, "Already-fixed sources were not preserved"
        source.write_text("void unrelated_dispatch() {}\n")
        result = transform(False)
        assert "Unsupported GGML CUDA dispatch" in result.stderr
        source.write_text(original + original)
        result = transform(False)
        assert "Unsupported GGML CUDA dispatch" in result.stderr
        source.write_text(original + first.decode())
        result = transform(False)
        assert "Mixed patched and unpatched" in result.stderr
        source.write_text(original)
        build = root / "build"
        result = subprocess.run([cmake, "-S", str(root), "-B", str(build)],
                                capture_output=True, text=True)
        assert result.returncode == 0, result.stdout + result.stderr
        assert source.read_text() == original
        sources = (build / "sources.txt").read_text().split(";")
        assert len(sources) == 2 and sources[0].endswith("/gliner-cuda/ggml-cuda.cu") and sources[1] == "other.cu", sources
        assert (build / "gliner-cuda" / "ggml-cuda.cu").read_bytes() == first
        assert root.as_posix() in (build / "includes.txt").read_text()
    print("CUDA precision source overlay is repeatable and leaves source checkouts unchanged")


if __name__ == "__main__":
    main(*sys.argv[1:])
