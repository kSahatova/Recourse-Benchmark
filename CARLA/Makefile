.PHONY: test clean install-dev requirements venv

#################################################################################
# GLOBALS                                                                       #
#################################################################################
PACKAGE_NAME = carla
PYTHON = $(VENV_DIR)/Scripts/python.exe
PIP = $(VENV_DIR)/Scripts/pip3.exe
VENV_DIR = ../venv

CURRENT_DIR = $(shell pwd)
#################################################################################
# COMMANDS                                                                      #
#################################################################################
## Run package


## Run python tests
##test:
##	$(PYTHON) -m pytest test/*


## Delete all compiled Python files
##clean:
##	find . -type f -name "*.py[co]" -delete
##	find . -type d -name "__pycache__" -delete

## Install development tools
install-dev: requirements
	$(VENV_DIR)/Scripts/pre-commit install


## Install Python Dependencies
requirements: venv
	$(PIP) install -U setuptools wheel
ifneq ($(wildcard ./setup.py),)
	$(PIP) install -e .
endif
ifneq ($(wildcard ./requirements.txt),)
	$(PIP) install -r requirements.txt
endif
ifneq ($(wildcard ./requirements-dev.txt),)
	$(PIP) install -r requirements-dev.txt
endif

## Install virtual environment
venv:
ifeq ($(wildcard $(VENV_DIR)/*),)
	mkdir -p $(VENV_DIR)
	python3.7 -m venv $(VENV_DIR)
endif
