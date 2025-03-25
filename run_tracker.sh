#!/bin/bash

DIRECTORY=$1

if [ -d "$TARGET_DIR" ]; then
    python -m src.tracker --dir $DIRECTORY --detect_all $@
else
    python -m src.tracker --dir ./data $@
fi