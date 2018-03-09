#!/usr/bin/env bash
python testCifar10.py -g 0.8 -m 0

python testCifar10_DropConnect.py -g 0.05 -m 1
python testCifar10_DropConnect.py -g 0.1 -m 1
python testCifar10_DropConnect.py -g 0.2 -m 1
python testCifar10_DropConnect.py -g 0.4 -m 1
python testCifar10_DropConnect.py -g 0.8 -m 1
