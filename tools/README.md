# Preset generation

`build_presets.py` generates the standalone `core10` and `cnn30` generation configs. It is used when changing presets and by CI to detect drift. Normal installation and execution do not require it.

```bash
python tools/build_presets.py --check
```

For an intentional preset change, edit the definitions in `build_presets.py`, run it with `--write`, and review the generated YAML and preset contract fixture together. Release evaluation configs refer to supplied ONNX candidates and are separate from these generation presets.
