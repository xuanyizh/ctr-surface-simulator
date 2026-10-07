#!/bin/sh
cd "$(dirname "$0")" || exit 1
python3 -m streamlit run app.py --server.address 127.0.0.1
