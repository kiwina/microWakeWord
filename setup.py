import setuptools

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setuptools.setup(
    name="microwakeword",
    version="0.1.0",
    install_requires=[
        "audiomentations", # Version managed by consuming project or ensure general compatibility
        # "audio_metadata", # To be removed/replaced
        "datasets",       # Version managed by consuming project
        "mmap_ninja",
        "numpy>=1.21",    # TF 2.19 base image has 2.1.3; ensure compatibility
        # "pymicro-features", # Assuming this is installed locally from a fork
        "pyyaml>=6.0",    # Base image has 6.0.2
        "tensorflow>=2.19.0,<2.20.0", # Align with base image
        "webrtcvad",
    ],
    author="Kevin Ahrendt",
    author_email="kahrendt@gmail.com",
    description="A TensorFlow based wake word detection training framework using synthetic sample generation suitable for certain microcontrollers.",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/kahrendt/microWakeWord",
    project_urls={
        "Bug Tracker": "https://github.com/kahrendt/microWakeWord/issues",
    },
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: Apache 2.0 License",
        "Operating System :: OS Independent",
    ],
    packages=setuptools.find_packages(),
    include_package_data=True,
    python_requires=">=3.11",
)
