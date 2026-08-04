import ast
import base64
from io import BytesIO
import json
from pathlib import Path

from PIL import Image


ROOT = Path(__file__).resolve().parents[1]
NOTEBOOK = ROOT / "examples" / "black_hole_colab.ipynb"
PREVIEW = NOTEBOOK.parent / "colab_preview.png"
COLAB_URL = (
    "https://colab.research.google.com/github/IakOBiaN/black-hole/"
    "blob/main/examples/black_hole_colab.ipynb"
)


def test_colab_notebook_has_valid_python_cells():
    notebook = json.loads(NOTEBOOK.read_text(encoding="utf-8"))
    assert notebook["nbformat"] == 4

    code_cells = [
        "".join(cell["source"])
        for cell in notebook["cells"]
        if cell["cell_type"] == "code"
    ]
    assert code_cells
    for source in code_cells:
        ast.parse(source)

    combined_source = "\n".join(code_cells)
    assert "https://github.com/IakOBiaN/black-hole.git" in combined_source
    assert "from black_hole.snapshot import main" in combined_source
    assert 'repo_dir / "out" / "colab.png"' in combined_source

    with Image.open(PREVIEW) as preview:
        assert preview.size == (720, 480)

    attachment = notebook["cells"][0]["attachments"]["colab_preview.png"]
    with Image.open(BytesIO(base64.b64decode(attachment["image/png"]))) as image:
        assert image.size == (720, 480)

    assert any(
        output.get("output_type") == "display_data"
        and "image/png" in output.get("data", {})
        for cell in notebook["cells"]
        for output in cell.get("outputs", [])
    )


def test_readme_links_to_colab_notebook():
    readme = (ROOT / "README.md").read_text(encoding="utf-8")
    assert COLAB_URL in readme
