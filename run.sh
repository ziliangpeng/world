#!/bin/bash
cd "$(dirname "$0")/.server"
source .venv/bin/activate
python server2.py
