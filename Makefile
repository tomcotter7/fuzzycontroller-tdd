test:
	pytest -vs --durations=0
reqs:
	pip freeze > requirements.txt
api-doc:
	sphinx-apidoc -o docs/source/ fuzzycontroller/ fuzzycontroller/**/tests/
clean-doc:
	rm -r docs/build/*
