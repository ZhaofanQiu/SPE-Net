#!/bin/bash

# compile custom operators
rm -r pytorchpoints/functional/pt_custom_ops/pt_custom_ops.egg-info/*
rm -r pytorchpoints/functional/pt_custom_ops/dist/*
rm -r pytorchpoints/functional/pt_custom_ops/build/*
rm -r pytorchpoints/functional/cpp_wrappers/cpp_subsampling/build/*
rm -r pytorchpoints/functional/cpp_wrappers/cpp_subsampling/*.so

cd pytorchpoints/functional/cpp_wrappers
sh compile_wrappers.sh
cd ../pt_custom_ops
python3 setup.py install --user
cd ../../..
