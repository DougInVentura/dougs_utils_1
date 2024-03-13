from setuptools import setup, find_packages

setup(
    name="dougs_utils_1",
    version="0.1",
    packages=find_packages(),
    install_requires=['scikit-learn','pandas'],
    # general package to house functions I want to reuse.
    author="Doug Francis",
    author_email="douginventura@gmail.com",
    description="right now, just has calc_metrics, which will calculate metrics for most classification models",
    license="None_as_yet",
    keywords="metrics",
)