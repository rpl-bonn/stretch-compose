[build-system]
requires = ["setuptools"]
build-backend = "setuptools.build_meta"

[project]
name = "dynagraph"
version = "0.0.1"
description = "A simple python package"
authors = [
  { name = "Rohit Menon" }
]
readme = "README.md"
requires-python = ">=3.8"
dependencies = []

[tool.setuptools.packages.find]
where = ["src"]
