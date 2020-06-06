import pathlib
from setuptools import setup, find_packages

# The directory containing this file
HERE = pathlib.Path(__file__).parent

# The text of the README file
README = (HERE / "README.md").read_text()

# This call to setup() does all the work
setup(
    name="revarie",
    version="1.3.9",
    description="Variogram manipulation and field generation toolbox",
    long_description=README,
    long_description_content_type="text/markdown",
    url="https://github.com/deanrp2/revarie",
    author="Dean Price",
    author_email="deanrp@umich.edu",
    license="MIT",
    classifiers=[
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.7",
    ],
    packages=find_packages(exclude=("tests",)),
    include_package_data=True,
    install_requires=["numpy_indexed"],
)
