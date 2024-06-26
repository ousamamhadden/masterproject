.PHONY: create_environment requirements dev_requirements clean data build_documentation serve_documentation

#################################################################################
# GLOBALS                                                                       #
#################################################################################

PROJECT_NAME = src
PYTHON_VERSION = 3.11.5
PYTHON_INTERPRETER = python

#################################################################################
# COMMANDS                                                                      #
#################################################################################

## Set up python interpreter environment
create_environment:
	conda create --name $(PROJECT_NAME) python=$(PYTHON_VERSION) --no-default-packages -y
	conda create --name cnn python=$(PYTHON_VERSION) --no-default-packages -y
	conda create --name vit python=$(PYTHON_VERSION) --no-default-packages -y

## Install Python Dependencies
requirements:
	$(PYTHON_INTERPRETER) -m pip install -U pip setuptools wheel
	$(PYTHON_INTERPRETER) -m pip install -r requirements.txt
	$(PYTHON_INTERPRETER) -m pip install -e .

requirements_cnn:
	$(PYTHON_INTERPRETER) -m pip install -U pip setuptools wheel
	$(PYTHON_INTERPRETER) -m pip install -r requirements_cnn.txt

requirements_vit:
	$(PYTHON_INTERPRETER) -m pip install -U pip setuptools wheel
	$(PYTHON_INTERPRETER) -m pip install -r requirements_vit.txt

## Install Developer Python Dependencies
dev_requirements: requirements
	$(PYTHON_INTERPRETER) -m pip install .["dev"]

## Delete all compiled Python files
clean:
	find . -type f -name "*.py[co]" -delete
	find . -type d -name "__pycache__" -delete


#################################################################################
# PROJECT RULES                                                                 #
#################################################################################

## Process raw data into processed data
data:
	python $(PROJECT_NAME)/data/make_dataset.py

## Make train model
.PHONY: train
train: requirements
	$(PYTHON_INTERPRETER) $(PROJECT_NAME)/train_model.py

## Make train model
.PHONY: train_cnn
train_cnn: requirements_cnn
	$(PYTHON_INTERPRETER) $(PROJECT_NAME)/train_cnn_model.py

.PHONY: train_cnn_extra_layers
train_cnn_extra_layers: requirements_cnn
	$(PYTHON_INTERPRETER) $(PROJECT_NAME)/train_cnn_model_extra_layers.py

.PHONY: train_cnn_large_filter
train_cnn_large_filter: requirements_cnn
	$(PYTHON_INTERPRETER) $(PROJECT_NAME)/train_cnn_model_large_filter.py

.PHONY: train_cnn_4clases
train_cnn_4classes: requirements_cnn
	$(PYTHON_INTERPRETER) $(PROJECT_NAME)/train_cnn_model_4classes.py

.PHONY: train_vit
train_vit: requirements_cnn
	$(PYTHON_INTERPRETER) $(PROJECT_NAME)/train_vit_model.py

## Use model to predict
.PHONY: predict
predict: requirements
	$(PYTHON_INTERPRETER) $(PROJECT_NAME)/predict_model.py $(model) $(data)
#Example: make predict  model=models/trained_model.pt data=data/processed/test_set.pt

#################################################################################
# Documentation RULES                                                           #
#################################################################################

## Build documentation
build_documentation: dev_requirements
	mkdocs build --config-file docs/mkdocs.yaml --site-dir build

## Serve documentation
serve_documentation: dev_requirements
	mkdocs serve --config-file docs/mkdocs.yaml

#################################################################################
# Self Documenting Commands                                                     #
#################################################################################

.DEFAULT_GOAL := help

# Inspired by <http://marmelab.com/blog/2016/02/29/auto-documented-makefile.html>
# sed script explained:
# /^##/:
# 	* save line in hold space
# 	* purge line
# 	* Loop:
# 		* append newline + line to hold space
# 		* go to next line
# 		* if line starts with doc comment, strip comment character off and loop
# 	* remove target prerequisites
# 	* append hold space (+ newline) to line
# 	* replace newline plus comments by `---`
# 	* print line
# Separate expressions are necessary because labels cannot be delimited by
# semicolon; see <http://stackoverflow.com/a/11799865/1968>
.PHONY: help
help:
	@echo "$$(tput bold)Available commands:$$(tput sgr0)"
	@sed -n -e "/^## / { \
		h; \
		s/.*//; \
		:doc" \
		-e "H; \
		n; \
		s/^## //; \
		t doc" \
		-e "s/:.*//; \
		G; \
		s/\\n## /---/; \
		s/\\n/ /g; \
		p; \
	}" ${MAKEFILE_LIST} \
	| awk -F '---' \
		-v ncol=$$(tput cols) \
		-v indent=19 \
		-v col_on="$$(tput setaf 6)" \
		-v col_off="$$(tput sgr0)" \
	'{ \
		printf "%s%*s%s ", col_on, -indent, $$1, col_off; \
		n = split($$2, words, " "); \
		line_length = ncol - indent; \
		for (i = 1; i <= n; i++) { \
			line_length -= length(words[i]) + 1; \
			if (line_length <= 0) { \
				line_length = ncol - indent - length(words[i]) - 1; \
				printf "\n%*s ", -indent, " "; \
			} \
			printf "%s ", words[i]; \
		} \
		printf "\n"; \
	}' \
	| more $(shell test $(shell uname) = Darwin && echo '--no-init --raw-control-chars')