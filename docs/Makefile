# Minimal makefile for Sphinx documentation
#

# You can set these variables from the command line.
SPHINXOPTS    =
SPHINXBUILD   = sphinx-build
SPHINXPROJ    = pliers
SOURCEDIR     = .
BUILDDIR      = _build

.PHONY: help clean Makefile

# Put it first so that "make" without argument is like "make help".
help:
	@$(SPHINXBUILD) -M help "$(SOURCEDIR)" "$(BUILDDIR)" $(SPHINXOPTS) $(O)

clean:
	rm -rf $(BUILDDIR)/*
	rm -rf generated/*

install:
	cd ..
	git checkout gh-pages
	# Guard against invocation from other directories
	rm -rf ../pliers/*
	# git clone --no-checkout --depth 1 https://github.com/tyarkoni/pliers.git docs
	git checkout sphinx-docs docs
	cd docs
	make clean
	make html
	cd ..
	cp -R docs/_build/html .
	rm -rf docs
	touch .nojekyll
	git add *
	git commit -m "installing sphinx docs"
	git push origin gh-pages

# Catch-all target: route all unknown targets to Sphinx using the new
# "make mode" option.  $(O) is meant as a shortcut for $(SPHINXOPTS).
%: Makefile
	@$(SPHINXBUILD) -M $@ "$(SOURCEDIR)" "$(BUILDDIR)" $(SPHINXOPTS) $(O)