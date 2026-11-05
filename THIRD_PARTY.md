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
