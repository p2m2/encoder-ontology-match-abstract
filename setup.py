from setuptools import setup, find_packages

setup(
    name="LLMSemanticAnnotator",
    version="0.1",
    packages=find_packages(),
    install_requires=[],
    entry_points={
        'console_scripts': [
            'llm-semantic-annotator=llm_semantic_annotator.__main__:main',
        ],
    },
    # Autres métadonnées
    author="Olivier Filangi",
    author_email="olivier.filangi@inrae.fr",
    description="Annotation sémantique de texte",
    long_description=open('README.md').read(),
    long_description_content_type="text/markdown",
    url="https://github.com/votre-nom/llm-semantic-annotator",
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.6',
)
