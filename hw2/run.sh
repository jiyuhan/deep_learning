#!/usr/bin/env bash

THEANO_FLAGS=device=cuda python testCifar10.py -g 0.8 -m 0

THEANO_FLAGS=device=cuda python testCifar10.py -g 0.05 -m 1
THEANO_FLAGS=device=cuda python testCifar10.py -g 0.1 -m 1
THEANO_FLAGS=device=cuda python testCifar10.py -g 0.2 -m 1
THEANO_FLAGS=device=cuda python testCifar10.py -g 0.4 -m 1
THEANO_FLAGS=device=cuda python testCifar10.py -g 0.8 -m 1
