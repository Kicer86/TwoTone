import sys
import re

SUMMARY_RE = re.compile(r"^Found \d+ errors? in \d+ files? \(")

def read_blocks(path):
    with open(path, "r", encoding="utf-8") as f:
        block = []
        for line in f:
            line = line.rstrip()
            if SUMMARY_RE.match(line):
                continue  # skip final summary line
            if ": error:" in line:
                if block:
                    yield block
                block = [line]
            elif block:
                block.append(line)
        if block:
            yield block

def block_key(block):
    return block[0]

if __name__ == "__main__":
    blocks = sorted(read_blocks(sys.argv[1]), key=block_key)
    for block in blocks:
        for line in block:
            print(line)
