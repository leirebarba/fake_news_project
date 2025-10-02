from setuptools import setup, find_packages

setup(
    name='fake_news_project',
    version='0.1.0',  # Starting version
    description='Analysis on whether probabilistic models or deep learning models are better at detecting fake vs real news',

    packages=find_packages("src"),
    package_dir={"":"src"}, 

)
