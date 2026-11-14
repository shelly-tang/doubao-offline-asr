#!/bin/zsh
cd -- "$(dirname -- "$0")"
/usr/bin/open http://localhost:8765
python3 web_server.py
