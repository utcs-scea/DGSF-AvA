#!/bin/bash -e

sudo apt update
sudo apt install -y lsb-release
if [[ $(lsb_release -rs) != "18.04" ]]; then
  echo "The support of $(lsb_release -ds) is untested. Continue (y/n)?"
  read -r yn_value
  if [[ ${yn_value} != "y" ]]; then
    echo "Dependency installation cancelled"
    exit 0
  fi
fi

sudo apt install -y apt-transport-https ca-certificates gnupg software-properties-common wget
wget -O - https://apt.kitware.com/keys/kitware-archive-latest.asc 2>/dev/null | gpg --dearmor - | sudo tee /etc/apt/trusted.gpg.d/kitware.gpg >/dev/null
sudo apt-add-repository 'deb https://apt.kitware.com/ubuntu/ bionic main'
sudo apt update
sudo apt purge --auto-remove cmake
sudo apt install -y cmake cmake-curses-gui
sudo apt install -y git build-essential python3 python3-pip libglib2.0-dev clang-7 libclang-7-dev \
  indent ninja-build
python3 -m pip install pip
python3 -m pip install setuptools pkgconfig
python3 -m pip install wget toposort astor 'numpy==1.15.0' blessings meson==0.58.1
