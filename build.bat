pyinstaller --noconfirm --log-level=WARN ^
    --onefile --noconsole ^
    --name MagiDraw ^
    ./src/main.py ^
    --add-data=./img/snake.pdf;./img ^
    --add-data=./img/dog.pdf;./img
