"""Install the bundled OpenPI patches into the active environment's Transformers."""
import pathlib
import shutil
import transformers


def main():
    if transformers.__version__ != "4.53.2":
        raise RuntimeError("The bundled patches require transformers==4.53.2")
    source = pathlib.Path(__file__).resolve().parents[1] / "src/openpi/models_pytorch/transformers_replace"
    destination = pathlib.Path(transformers.__file__).resolve().parent
    for path in source.rglob("*.py"):
        target = destination / path.relative_to(source)
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(path, target)
    print("Installed OpenPI Transformers patches. Restart Python before importing the evaluator.")


if __name__ == "__main__":
    main()
