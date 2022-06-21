import io
import os
import subprocess
import sys

import setuptools

try:
    from numpy import get_include
except ImportError:
    subprocess.check_call([sys.executable, "-m", "pip", "install", "numpy"])
    from numpy import get_include

# Package meta-data.
NAME = "TSEvo"
DESCRIPTION = "todo"
LONG_DESCRIPTION_CONTENT_TYPE = "text/markdown"
URL = "https://github.com/JHoelli/TSEvo"
EMAIL = "hoellig@fzi.de"
AUTHOR = "Jacqueline Hoellig"
REQUIRES_PYTHON = ">=3.6.0"

# Package requirements.
base_packages = [
    "h5py", # todo add version
    "joblib==1.0.1",
    "kaggle==1.5.12",
    "lime==0.2.0.1",
    "Markdown==3.3.4",
    "matplotlib==3.3.4",
    "pandas==1.1.5",
    "partd==1.2.0",
    "pytz==2021.3",
    "scikit-learn==0.24.2",
    "shap==0.39.0",
    "sktime==0.8.0",
    "tensorflow",#TODO used to be 2.6.0.
    "keras",#TODO used to be 2.6
    "tqdm==4.62.3",
    "tsfresh==0.18.0",
    "tslearn",
    "torch",
    "seaborn",
    "scikit_optimize",
    "mlrose",
    "torchcam",
    "tf_explain",
    "opencv-python",
    "captum",
    "pyts",
    "deprecated"
    
]

dev_packages = base_packages + [
    "mypy>=0.761",
    "pre-commit>=2.9.2",
    "pytest>=4.5.0",
    "pytest-cov>=2.6.1",
]

docs_packages = [
    "flask==2.0.2",
    "ipykernel==6.9.0",
    "mike==0.5.3",
    "mkdocs==1.2.3",
    "mkdocs-awesome-pages-plugin==2.7.0",
    "mkdocs-material==8.1.11",
    "mkdocstrings==0.18.0",
    "mkdocs-material-extensions",
    "mkdocs-autorefs",
    "ipython_genutils==0.1.0",
    "mkdocs-jupyter==0.20.0",
    "nbconvert==6.4.2",
    "numpydoc==1.2",
    "spacy==3.2.2",
    "jinja2==3.0.3",
]

here = os.path.abspath(os.path.dirname(__file__))

# Import the README and use it as the long-description.
with io.open(os.path.join(here, "README.md"), encoding="utf-8") as f:
    long_description = "\n" + f.read()

# Load the package's __version__.py module as a dictionary.
about = {}
with open(os.path.join(here, NAME, "__version__.py")) as f:
    exec(f.read(), about)

# Where the magic happens:
setuptools.setup(
    name=NAME,
    version=about["__version__"],
    description=DESCRIPTION,
    long_description=long_description,
    long_description_content_type=LONG_DESCRIPTION_CONTENT_TYPE,
    author=AUTHOR,
    author_email=EMAIL,
    python_requires=REQUIRES_PYTHON,
    url=URL,
    packages=setuptools.find_packages(exclude=("tests",)),
    install_requires=base_packages,
    extras_require={
        "dev": dev_packages,
        "test": dev_packages,
        "docs": docs_packages,
        "all": dev_packages,# + docs_packages,
        ":python_version == '3.6'": ["dataclasses"],
    },
    include_package_data=True,
    license="BSD-3",
    classifiers=[
        # Trove classifiers
        # Full list: https://pypi.python.org/pypi?%3Aaction=list_classifiers
        "License :: OSI Approved :: BSD License",
        "Programming Language :: Python",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.6",
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: Implementation :: CPython",
        "Programming Language :: Python :: Implementation :: PyPy",
    ],
    ext_modules=[]
)