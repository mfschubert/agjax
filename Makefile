
install:
	pip install -e .[dev]
	pre-commit install

dev:
	pip install -e .[dev,docs]

test:
	pytest -s

cov:
	pytest --cov=agjax

mypy:
	mypy . --ignore-missing-imports

pylint:
	pylint agjax

ruff:
	ruff --fix src/agjax/*.py
	ruff --fix src/agjax/experimental/*.py

git-rm-merged:
	git branch -D `git branch --merged | grep -v \* | xargs`

update:
	pur

update-pre:
	pre-commit autoupdate --bleeding-edge

release:
	git push
	git push origin --tags

build:
	rm -rf dist
	pip install build
	python -m build

docs:
	jb build docs

.PHONY: drc doc docs
