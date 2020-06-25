# PLAY WITH ML (An Intuitive way of implementing ML models).

## Index
  * [Demo](#demo)
  * [Overview](#overview)
  * [Installation](#installation)
  * [Directory Tree](#directory-tree)
  * [Use](#use)
  * [Deployement on Heroku](#deployement-on-heroku)
  * [To Do](#to-do)
  * [Bug / Feature Request](#bug---feature-request)
  * [Technologies Used](#technologies-used)
  * [Team](#team)
  * [License](#license)


## Demo
Link: [http://playwithml.herokuapp.com/](http://playwithml.herokuapp.com/)

[![](https://imgur.com/gec0n1K.png)](http://playwithml.herokuapp.com/)

## Overview
* This is a intuitive way of implementing Machine Learning models without writing even a single line of code.
* This is a supervised Machine Learning based Project in which you can Upload a Dataset of your choice,
* View the Charts and Bar graphs related to the Dataset,
* Select the type of model (Classification or Regression),
* Select which model to train on,
* Click on the TRAIN button,
* Finally get the Scores, Code and Report for your Model :)

## Installation
The Code is written in Python 3.7. If you don't have Python installed you can find
it [here](https://www.python.org/downloads/).
If you are using a lower version of Python you can upgrade using the pip package,
ensuring you have the latest version of pip. 
To install the required packages and libraries, run this command in the project 
directory:
```bash
git clone https://github.com/mystery2828/playwithML.git
pip install -r requirements.txt
```

## Directory Tree
```
│   app.py
│   classification.py
│   LICENSE
│   Procfile
│   readme.txt
│   regressor.py
│   requirements.txt
│   setup.sh
│
├───datasets
│       abalone.csv
│       Admission_Predict.csv
│       ionosphere.csv
│       iris.csv
│       Placement_Data_Full_Class.csv
│       pulse.xlsx
│       sonar.csv
│       winequality.csv
```

## Use
```bash
streamlit run app.py
```
[![](https://imgur.com/MtvTsL1.png)](http://playwithml.herokuapp.com/)
[![](https://imgur.com/x3ct0ou.png)](http://playwithml.herokuapp.com/)
[![](https://imgur.com/LkxcPJK.png)](http://playwithml.herokuapp.com/)
[![](https://imgur.com/CiVrqNu.png)](http://playwithml.herokuapp.com/)
[![](https://imgur.com/gzFq0GJ.png)](http://playwithml.herokuapp.com/)
[![](https://imgur.com/CiVrqNu.png)](http://playwithml.herokuapp.com/)

## Deployement on Heroku
```bash
git init
git add .
git commit -m "< commit message >"
git push
git push heroku master
```

## Bug / Feature Request
If you find a bug (the website couldn't handle the query and / or gave undesired results), kindly open an issue [here](https://github.com/mystery2828/playwithml/issues/new) by including your search query and the expected result.

## Technologies Used

![](https://forthebadge.com/images/badges/made-with-python.svg)

[<img target="_blank" src="https://images.g2crowd.com/uploads/product/image/social_landscape/social_landscape_77c883b19775c25838d2055fc2e7387e/scikit-learn.png" width=300>](https://scikit-learn.org/stable/) 
[<img target="_blank" src="https://pbs.twimg.com/profile_images/1234856290058428416/8lWJhqj1_400x400.jpg" width=170>](https://www.streamlit.io/) 
[<img target="_blank" src="https://miro.medium.com/max/1080/1*_oSOImPmBFeKj8vqE4FCkQ.jpeg" width=280>](https://pandas.pydata.org/)
[<img target="_blank" src="https://buddy.works/guides/thumbnails/cover-heroku.png" width=200>](https://www.heroku.com/)

## Team
[![Akash C](https://avatars1.githubusercontent.com/u/40836377?s=144&u=884f530d1deeb1897ccb6f83cea9e84cc3de4b28&v=4)](https://www.linkedin.com/in/akash-c-3a0468148/) |
-|
[Akash C](https://www.linkedin.com/in/akash-c-3a0468148/) |
[![Ashwin Sharma](https://avatars0.githubusercontent.com/u/51113630?s=144&u=a16f967611c067eb66f36bc1c070fe8fbe1d5341&v=4&s=144)](https://www.linkedin.com/in/ashwinsharmap/) |
-|
[Ashwin Sharma](https://www.linkedin.com/in/ashwinsharmap/)

## License
[![Apache license](https://img.shields.io/badge/license-apache-blue?style=for-the-badge&logo=appveyor)](http://www.apache.org/licenses/LICENSE-2.0e)

Copyright 2020 Akash C

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
