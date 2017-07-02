push:
	git add -A && git commit -m "auto commit" && git push origin master --force

list:
	python3 list2md.multiprocess.py
