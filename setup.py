from setuptools import setup

setup(
   name='spensum',
   version='0.0.1',
   description='Energy based neural network summarizer ',
   author='Chris Kedzie',
   author_email='kedzie@cs.colubmia.edu',
   packages=[],  #same as name
   dependency_links = [
       'git+https://github.com/kedz/simple_cnlp.git#egg=simple_cnlp',
       'git+https://github.com/kedz/duc_preprocess.git#egg=duc_preprocess',
       'git+https://github.com/kedz/rouge_papier.git#egg=rouge_papier',],
   install_requires = [
       'simple_cnlp',
       'duc_preprocess',
       'rouge_papier'],
)
