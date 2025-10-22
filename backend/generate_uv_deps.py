import os
import ast
from pathlib import Path
import subprocess

def find_imports(project_dir: str):
    imports = set()
    for root, _, files in os.walk(project_dir):
        for file in files:
            if file.endswith(".py"):
                filepath = os.path.join(root, file)
                try:
                    with open(filepath, "r", encoding="utf-8") as f:
                        node = ast.parse(f.read(), filename=file)
                        for n in ast.walk(node):
                            if isinstance(n, ast.Import):
                                for alias in n.names:
                                    imports.add(alias.name.split('.')[0])
                            elif isinstance(n, ast.ImportFrom):
                                if n.module:
                                    imports.add(n.module.split('.')[0])
                except Exception:
                    pass
    return imports


def filter_stdlib(imports):
    try:
        stdlib = subprocess.run(
            ["python3", "-m", "pydoc", "modules"],
            capture_output=True,
            text=True,
        ).stdout.splitlines()
        stdlib = {line.strip().split(" ")[0] for line in stdlib if line.strip()}
    except Exception:
        stdlib = set()
    return {pkg for pkg in imports if pkg not in stdlib and pkg not in ("__future__",)}


def generate_pyproject(imports, project_dir):
    pyproject_path = Path(project_dir) / "pyproject.toml"
    deps = sorted(list(imports))

    print(f"\n🔍 Detected {len(deps)} dependencies:")
    for d in deps:
        print(f" - {d}")

    deps_list = ",\n    ".join([f'"{dep}"' for dep in deps])
    template = f"""[project]
name = "my-project"
version = "0.1.0"
description = "Auto-generated dependencies"
requires-python = ">=3.9"
dependencies = [
    {deps_list}
]
"""

    pyproject_path.write_text(template, encoding="utf-8")
    print(f"\n✅ pyproject.toml generated at: {pyproject_path}")


if __name__ == "__main__":
    project_dir = "."
    all_imports = find_imports(project_dir)
    deps = filter_stdlib(all_imports)
    generate_pyproject(deps, project_dir)

