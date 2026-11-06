# Preset generation

`build_presets.py` generates the standalone `core10` and `cnn30` generation configs. It is used when changing presets and by CI to detect drift. Normal installation and execution do not require it.

```bash
python tools/build_presets.py --check
```

For an intentional preset change, edit the definitions in `build_presets.py`, run it with `--write`, and review the generated YAML and preset contract fixture together. Release evaluation configs refer to supplied ONNX candidates and are separate from these generation presets.

## Fixed-input Release conversion

`make_fixed_spatial_bundle.py` creates a new catalog and models from a verified CNN30 catalog. It preserves all candidates and default selections. Five models receive fixed network input dimensions and the packaged input policies. Other networks and CPU host graphs are copied unchanged. Existing output directories are rejected.

```bash
python tools/make_fixed_spatial_bundle.py original-bundle/catalog.json static-bundle
```

Use `static-bundle/configs/core10.yaml`, `cnn30.yaml` or `core10-candidates.yaml` with `run -c`. Conversion does not recalibrate or train models. Evaluate FP32 and INT8 with the same new input policy before comparing recovery. The source checkout and its runtime dependencies are required for this maintenance tool.
