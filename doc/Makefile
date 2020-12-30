# Minimal makefile for Sphinx documentation
#

# You can set these variables from the command line, and also
# from the environment for the first two.
SPHINXOPTS    ?=
SPHINXBUILD   ?= sphinx-build
SOURCEDIR     = .
BUILDDIR      = _build

# Put it first so that "make" without argument is like "make help".
help:
	@$(SPHINXBUILD) -M help "$(SOURCEDIR)" "$(BUILDDIR)" $(SPHINXOPTS) $(O)

.PHONY: help Makefile

# Catch-all target: route all unknown targets to Sphinx using the new
# "make mode" option.  $(O) is meant as a shortcut for $(SPHINXOPTS).
%: Makefile
	@$(SPHINXBUILD) -M $@ "$(SOURCEDIR)" "$(BUILDDIR)" $(SPHINXOPTS) $(O)

clean:
	rm -rf $(BUILDDIR)/*
	rm -rf auto_examples
	rm -rf auto_tutorials
	rm -rf auto_motivations
	rm -rf generated


install:
	# Clean out existing build
	make clean
	# Clone, specifically, the gh-pages branch, putting it into '_build/gh_pages/'
	git clone -b gh-pages --single-branch --no-checkout --depth 1 https://github.com/voytekresearch/ndspflow _build/gh_pages
	# A .nojekyll file tells Github pages to bypass Jekyll processing
	touch _build/gh_pages/.nojekyll
	# Build the sphinx site
	make html
	# Create example reports
	ndspflow \
	-freqs freqs.npy \
	-power_spectrum powers.npy \
	-sig sigs.npy \
	-fs 500 \
	-f_range_fooof 1 50 \
	-f_range_bycycle 15 25 \
	-max_n_peaks 1 \
	-min_peak_height .3 \
	-peak_threshold 2 \
	-peak_width_limits 1 5 \
	-aperiodic_mode fixed \
	-center_extrema peak \
	-burst_method cycles \
	-amp_fraction_threshold .25 \
	-amp_consistency_threshold 0 \
	-period_consistency_threshold .5 \
	-monotonicity_threshold .8 \
	-min_n_cycles 3 \
	-axis 0 \
	-n_jobs -1 \
	-run_nodes both \
	$(PWD)/tutorials/data $(PWD)/_build/html/results
	# Ensure js is available and convert local paths
	mkdir -p $(PWD)/_build/html/resources/js $(PWD)/_build/html/resources/css
	cp $(PWD)/../ndspflow/reports/templates/resources/js/* $(PWD)/_build/html/resources/js/
	cp $(PWD)/../ndspflow/reports/templates/resources/css/* $(PWD)/_build/html/resources/css/
	cp $(PWD)/_build/html/results/bycycle/.data.js $(PWD)/_build/html/resources/js/
	for f in $$(find _build/html/results -name "*.html"); do \
	    sed -i -e "s#src=.*bycycle/#src=\"\ndspflow/resources/js/#" $$f; \
	    sed -i -e "s#src=.*js/#src=\"/ndspflow/resources/js/#" $$f; \
	    sed -i -e "s#href=.*css/#href=\"/ndspflow/resources/css/#" $$f; \
 	    sed -i -e "s#href=.*/_build/html#href=\"/ndspflow#g" $$f; \
	    sed -i -e "s#file:///[^_]*##g" -e "s#_build/html#/ndspflow#g"  $$f; \
	done
	# Copy site into the gh-pages branch folder, then push to Github to deploy
	cd _build/ && \
	cp -r html/* gh_pages && \
	cd gh_pages && \
	git add * && \
	git add .nojekyll && \
	git commit -a -m 'Make install' && \
	git push
