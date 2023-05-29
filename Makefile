test:
	pytest -vs --durations=0
reqs:
	pip freeze > requirements.txt
