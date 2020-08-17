import setuptools
with open("README.md", "r") as fh:
    long_description = fh.read()
setuptools.setup(
     name='playwithML',
     version='0.2',
     scripts=['playwithml.py'],
     author="Akash C",
     author_email="akashincrp@gmail.com",
     description="A AutoML type utility",
     long_description=long_description,
   long_description_content_type="text/markdown",
     url="https://github.com/mystery2828/playwithml",
     packages=setuptools.find_packages(),
     classifiers=[
         "Programming Language :: Python :: 3",
         "License :: OSI Approved :: MIT License",
         "Operating System :: OS Independent",
     ],
 )