#!/bin/bash
###################
sudo apt update
sudo apt install python3 python3-venv python3-pip -y
sudo apt install memcached libmemcached-tools -y
sudo systemctl enable memcached
sudo systemctl start memcached
###################
python3 -m venv venv
source venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
