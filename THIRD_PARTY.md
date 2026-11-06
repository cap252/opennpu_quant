# Third-party notices

opennpu-quant is licensed under [Apache-2.0](LICENSE). Attribution for included
third-party code and alignment conventions is collected in [NOTICE](NOTICE).
The corresponding license texts are packaged under
`opennpu_quant/resources/licenses/`.

Framework and export dependencies are installed separately and retain their own
license notices. Dependency versions are recorded in `pyproject.toml` and
`requirements/`. Model recipes retain upstream source URLs, revisions and
checkpoint hashes.

The Python package does not bundle upstream model repositories, trained weights
or datasets. Model Release archives contain converted pretrained weights and
their INT8 variants. The model assets retain their applicable license terms,
separately from this package's license. Datasets are not included in the archives.
Exporting or quantizing a model does not itself grant redistribution rights for
the resulting asset.

The [model-notice record for this release](https://github.com/cap252/opennpu_quant/blob/v1.0.2-static.1/MODEL-NOTICES.md)
describes CNN30 model sources, weight-license evidence and conversion steps.
Model archives for this release include `MODEL-NOTICES.md` and the original
notices under `licenses/models/`. The Python package's license does not replace
model-specific terms.
