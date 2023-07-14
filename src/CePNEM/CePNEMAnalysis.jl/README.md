# CePNEMAnalysis.jl

[![][docs-stable-img]][docs-stable-url] [![][docs-latest-img]][docs-latest-url]

[docs-stable-img]: https://img.shields.io/badge/docs-stable-blue.svg
[docs-stable-url]: https://flavell-lab.github.io/CePNEMAnalysis.jl/stable/

[docs-latest-img]: https://img.shields.io/badge/docs-latest-blue.svg
[docs-latest-url]: https://flavell-lab.github.io/CePNEMAnalysis.jl/dev/ 

This package provides a collection of tools for interpreting and visualizing CePNEM model fits. It comes with four companion notebooks (available in the `notebook` directory in this package):

- `CePNEM-analysis.ipynb`: Loads raw CePNEM fit data and computes metrics such as neural encoding, encoding change, variability, and more.
- `CePNEM-plots.ipynb`: Presents the most common plots used to visualize CePNEM fits, as well as a guide to exploring neural encodings generated from the `CePNEM-analysis.ipynb` notebook. This notebook can also be used by downloading our preprocessed data from wormwideweb.org and examining it here.
- `CePNEM-UMAP.ipynb`: Demonstrates how to use UMAP to visualize CePNEM fits.
- `CePNEM-auxiliary.ipynb`: Presents less-commonly used plots and functions, such as model validation metrics, decoder training, and more.

## Citation

Brain-wide representations of behavior spanning multiple timescales and states in *C. elegans*

Adam A. Atanas*, Jungsoo Kim*, Ziyu Wang, Eric Bueno, McCoy Becker, Di Kang, Jungyeon Park, Cassi Estrem, Talya S. Kramer, Saba Baskoylu, Vikash K. Mansinghka, Steven W. Flavell

bioRxiv 2022.11.11.516186; doi: https://doi.org/10.1101/2022.11.11.516186

\* equal contribution
