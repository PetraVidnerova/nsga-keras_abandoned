#!/usr/bin/env bash

apt-get update
# convenience tools
apt-get install -y \
    less \
    vim
# jupyter lab bin deps
apt-get install -y \
    nodejs \
    npm
apt-get clean
