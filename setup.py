from setuptools import find_packages, setup

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name="road-marking-vectorizer",
    version="0.3.0",
    description="Simplified road marking extractor (point cloud to DXF)",
    author="Road Marking Team",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    include_package_data=True,
    python_requires=">=3.10",
    install_requires=[
        "numpy>=1.23",
        "scipy>=1.9",
        "laspy>=2.4",
        "open3d>=0.17",
        "opencv-python>=4.7",
        "ezdxf>=1.1",
        "shapely>=2.0",
    ],
    entry_points={
        "console_scripts": [
            "rmc-run=road_marking_classifier.cli.main:main",
        ]
    },
    long_description=long_description,
    long_description_content_type="text/markdown",
)
