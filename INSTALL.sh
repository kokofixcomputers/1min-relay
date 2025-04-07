#!/bin/bash
###################
# run with sudo
apt update
apt install python3 python3-venv python3-pip -y
apt install memcached libmemcached-tools -y
systemctl enable memcached
systemctl start memcached
###################
python3 -m venv venv
source venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
