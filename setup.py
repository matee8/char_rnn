from setuptools import setup, find_packages

NAME = "char_rnn"
VERSION = "0.1.0"
DESCRIPTION = "A character-level recurrent neural network implementation."
AUTHOR = "matee8"
AUTHOR_EMAIL = "graves-bluff-pesky@duck.com"
URL = "https://github.com/matee8/char_rnn"

REQUIRED = ["numpy==2.2.5"]

try:
    with open("README.md", "r", encoding="utf-8") as f:
        LONG_DESCRIPTION = f.read()
except FileNotFoundError:
    LONG_DESCRIPTION = DESCRIPTION

setup(name=NAME,
      version=VERSION,
      description=DESCRIPTION,
      long_description=LONG_DESCRIPTION,
      long_description_content_type="text/markdown",
      author=AUTHOR,
      author_email=AUTHOR_EMAIL,
      url=URL,
      package_dir={"": "src"},
      packages=find_packages(where="src"),
      install_requires=REQUIRED)
