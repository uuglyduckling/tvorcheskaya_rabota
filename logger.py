import sys
from pathlib import Path

#Базовый логгер, всем знакомый, если работал тестировщиком -- всем советую, классный опыт
class Logger:
    def __init__(self, filepath: str):
        self.terminal = sys.stdout
        self.log = open(filepath, "w", encoding="utf-8")

    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)

    def flush(self):
        pass
