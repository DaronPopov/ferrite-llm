#!/bin/bash
# Build the spcpp runner for Linux
cd "$(dirname "$0")/.."

mkdir -p bin

g++ -O3 -std=c++17 src/spcpp_runner.cpp -o bin/spcpp

echo "Built: bin/spcpp"
