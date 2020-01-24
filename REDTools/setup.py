import setuptools

setuptools.setup(
    name="REDTools", # Replace with your own username
    version="0.0.1",
    author="Alex Anwyl-Irvine",
    author_email="Alexander.Irvine@mrc-cbu.cam.ac.uk",
    description="Utility Tools for the RED big-data neuroimaging project, mainly wrappers for MNE package",
    long_description="Utility Tools for the RED big-data neuroimaging project",
    long_description_content_type="text/markdown",
    url="https://github.com/u01ai11/REDTools",
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.6',
)