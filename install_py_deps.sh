#!/usr/bin/env bash

pip install --no-cache-dir -r requirements.txt

jupyter labextension install \
	@jupyter-widgets/jupyterlab-manager \
	jupyter-matplotlib
