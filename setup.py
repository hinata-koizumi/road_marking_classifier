from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

with open("requirements.txt", "r", encoding="utf-8") as fh:
    requirements = [line.strip() for line in fh if line.strip() and not line.startswith("#")]

setup(
    name="road-marking-classifier",
    version="1.0.0",
    author="Road Marking Classifier Project",
    author_email="contact@roadmarkingclassifier.com",
    description="Automated classification and color-coded output system for road markings from LiDAR point cloud data",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/yourusername/road-marking-classifier",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering :: GIS",
        "Topic :: Scientific/Engineering :: Image Processing",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.8",
    install_requires=requirements,
    entry_points={
        "console_scripts": [
            "road-marking-classifier=main:main",
        ],
    },
    keywords="lidar, point-cloud, road-marking, classification, dxf, computer-vision, transportation",
    project_urls={
        "Bug Reports": "https://github.com/yourusername/road-marking-classifier/issues",
        "Source": "https://github.com/yourusername/road-marking-classifier",
        "Documentation": "https://github.com/yourusername/road-marking-classifier/wiki",
    },
)