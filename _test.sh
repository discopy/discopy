#!/bin/bash
paths=$(
for module in $(while read line; do echo $line; done < to_test.txt)
do echo "*/*$module.py "
done)
coverage run -m pytest $paths --doctest-modules
coverage report -m $paths
