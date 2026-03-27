#!/bin/bash
apt-get update
apt-get install -y qemu-user-static
apt-get install -y cmake
apt-get install -y ninja-build
tar -xvf riscv64-glibc-ubuntu-24.04-gcc-nightly-2025.06.13-nightly.tar.xz
chmod 777 -R ncnn_with_demo/
chmod 777 -R riscv

