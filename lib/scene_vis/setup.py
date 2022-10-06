from setuptools import find_packages, setup

setup(
    name="scene-vis", packages=find_packages(where="src"), package_dir={"": "src"},
)
