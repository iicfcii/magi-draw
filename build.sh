#!/bin/bash
pyinstaller --noconfirm --log-level=WARN \
    --onefile --noconsole \
    --name MagiDraw \
    ./src/main.py \
    --add-data=./img/snake.pdf:./img \
    --add-data=./img/dog.pdf:./img \
    --add-data=./img/ball.pdf:./img

# Apply fix to the signle .app
# https://github.com/pyinstaller/pyinstaller/issues/3820#issuecomment-515673901
mkdir ./dist/MagiDraw.app/Contents/lib/
cp -R ./fix/tcl8 ./dist/MagiDraw.app/Contents/lib/tcl8
