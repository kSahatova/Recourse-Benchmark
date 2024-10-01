import pathlib
import re

from setuptools import find_packages, setup

VERSIONFILE = "carla/_version.py"
verstrline = open(VERSIONFILE, "rt").read()
VSRE = r"^__version__ = ['\"]([^'\"]*)['\"]"
mo = re.search(VSRE, verstrline, re.M)
if mo:
    VERSION = mo.group(1)
else:
    raise RuntimeError("Unable to find version string in %s." % (VERSIONFILE,))

# The directory containing this file
HERE = pathlib.Path(__file__).parent

# The text of the README file
README = (HERE / "README.md").read_text()

setup(
    name="carla-recourse",
    version=VERSION,
    description="A library for counterfactual recourse",
    long_description=README,
    long_description_content_type="text/markdown",
    url="https://github.com/carla-recourse/CARLA",
    author="Martin Pawelczyk, Sascha Bielawski, Johannes van den Heuvel, Tobias Richter and Gjergji Kasneci",
    author_email="martin.pawelczyk@uni-tuebingen.de",
    packages=["carla", "carla.data",
              "carla.evaluation", "carla.models",
              "carla.plotting", "carla.recourse_methods"],
    include_package_data=True
)
