import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="emtracks",
    version="0.0.0",
    author="Cole Kampa",
    author_email="ckampa13@gmail.com",
    description="Particle in EM fields track calculator",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/FMS-Mu2e/particle_EM_tracks",
    packages=setuptools.find_packages(),
    install_requires=[
        "numpy", "scipy", "pandas", "matplotlib", "pypdt", "dill"
    ],
    python_requires='>=3.6',
    include_package_data=True,
    zip_safe=False,)
