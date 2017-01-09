.PHONY: build coverage lint graphs tests docs clean clean_pyc clean_c clean_so clean_cython deploy

build:
	mkdir -p docs/modules/generated
	python setup.py build_ext -i -b .

build_cython:
	mkdir -p docs/modules/generated
	python setup.py build_ext -i cython

coverage:
	nosetests --logging-level=INFO --with-coverage --cover-package=discretize --cover-html
	open cover/index.html

lint:
	pylint --output-format=html discretize > pylint.html

graphs:
	pyreverse -my -A -o pdf -p discretize discretize/**.py discretize/**/**.py

tests:
	nosetests --logging-level=INFO

docs:
	cd docs;make html

clean_pyc:
	find . -name "*.pyc" | xargs -I {} rm -v "{}"

clean_c:
	find . -name "*.c" | xargs -I {} rm -v "{}"

clean_so:
	find . -name "*.so" | xargs -I {} rm -v "{}"

clean: clean_pyc
	cd docs;make clean

clean_cython: clean clean_c clean_so

deploy:
	python setup.py sdist bdist_wheel upload
