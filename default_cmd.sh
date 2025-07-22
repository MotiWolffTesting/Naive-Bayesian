#!/bin/sh
python main.py
# Keep the container running
exec tail -f /dev/null 