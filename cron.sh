#!/bin/sh
git pull origin master
python3 list2md.multiprocess.py
git commit -m "Auto update" -a
git push origin master
