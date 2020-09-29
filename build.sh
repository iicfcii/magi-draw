#!/bin/bash
pyinstaller --noconfirm --log-level=WARN \
    --onefile \
    --name MagiDraw \
    ./src/main.py
