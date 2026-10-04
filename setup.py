from setuptools import setup, find_packages

setup(
    name="razorkv",
    version="0.1.0",
    description="Extreme Dynamic KV Cache Sparsification & Layer-Pyramid Eviction for Long-Context LLMs",
    long_description=open("README.md", encoding="utf-8").read(),
    long_description_content_type="text/markdown",
    author="RazorKV Core Contributors",
    author_email="dev@razorkv.org",
    url="https://github.com/jdymitarai/razorkv",
    license="Apache-2.0",
    packages=find_packages(),
    python_requires=">=3.9",
    install_requires=[
        "torch>=2.0.0",
    ],
    extras_require={
        "all": ["transformers>=4.36.0", "triton>=2.1.0", "accelerate"],
        "benchmark": ["matplotlib", "tabulate", "tqdm"],
    },
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: Apache Software License",
        "Programming Language :: Python :: 3",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
)
