from setuptools import setup, find_packages

setup(
    name="steam-trend-analytics-uni",
    version="1.0.0",
    description="Eine empirische Analyse von Steam-Marktdaten mittels OLS-Regression",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    author="Constanze Baumgart & Daniel Schmidt",
    author_email="constanze.baumgart@fom-net.de",
    url="https://github.com/example/steam-analytics-uni",
    
    # Automatische Paketerkennung im src-Ordner
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    
    # Abhängigkeiten (Redundant zu requirements.txt, aber professionell)
    install_requires=[
        "pandas>=2.2.0",
        "numpy>=1.26.0",
        "streamlit>=1.31.0",
        "plotly>=5.19.0", 
        "statsmodels>=0.14.1",
        "kagglehub>=0.2.0"
    ],
    
    # Metadaten für Klassifizierung
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering :: Information Analysis",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    
    # Skripte, die ausführbar sein sollen
    entry_points={
        "console_scripts": [
            "run-dashboard=app:main",
        ],
    },
    
    python_requires=">=3.9",
)