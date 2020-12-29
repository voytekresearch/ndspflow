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

