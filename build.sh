#!/bin/bash
pyinstaller --noconfirm --log-level=WARN \
    --onefile --noconsole \
    --name MagiDraw \
    ./src/main.py

# Apply fix to the signle .app
# https://github.com/pyinstaller/pyinstaller/issues/3820#issuecomment-515673901
mkdir ./dist/MagiDraw.app/Contents/lib/
cp -R ./fix/tcl8 ./dist/MagiDraw.app/Contents/lib/tcl8
