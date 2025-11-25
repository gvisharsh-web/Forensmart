from setuptools import setup, find_packages

setup(
    name="forensmart",
    version="2.0.0",
    description="Professional Digital Forensic Analysis Platform",
    author="ForenSmart Team",
    packages=find_packages(),
    python_requires=">=3.9",
    install_requires=[
        "streamlit>=1.28.0",
        "pandas>=1.5.0",
        "numpy>=1.24.0",
        "plotly>=5.0.0",
        "openai>=1.0.0",
        "schedule>=1.2.0",
        "requests>=2.31.0",
    ],
)
