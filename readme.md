# SMS Spam Classifier
![made-with-python](https://img.shields.io/badge/Made%20with-Python-0078D4.svg)
![html5](https://img.shields.io/badge/HTML5-E34F26?logo=html5&logoColor=white)
![css3](https://img.shields.io/badge/CSS3-1572B6?logo=css3&logoColor=white)
![pandas](https://img.shields.io/badge/Pandas-2C2D72?logo=pandas&logoColor=white)
![scikit-learn](https://img.shields.io/badge/Scikit_learn-0078D4?logo=scikit-learn&logoColor=white)
![fastapi](https://img.shields.io/badge/Fastapi-109989?logo=FASTAPI&logoColor=white)
![vscode](https://img.shields.io/badge/Visual_Studio_Code-0078D4?logo=visual%20studio%20code&logoColor=white)

SMS Spam Classifier app is used to predict if the given input message is spam or not. The app created using python's scikit-learn, fastapi, pandas, nltk and joblib packages.

## Installation
Open Anaconda prompt and create new environment
```
conda create -n your_env_name python=(any version < 3.10.4)
```
Then Activate the newly created environment
```
conda activate your_env_name
```
Clone the repository using `git`
```
git clone https://github.com/Prakashdeveloper03/SMS-Spam-Classifier.git
```
Change to the cloned directory
```
cd <directory_name>
```
To install all requirement packages for the app
```
pip install -r requirements.txt
```
Then, Run the app
```
uvicorn main:app --reload
```
## 📷 Screenshots
### Spam Result
![spam_image](markdown/spam.png)
### Not Spam Result
![not_spam_image](markdown/ham.png)
### Swagger UI
![swagger_image](markdown/docs.png)
### Redoc UI
![redoc_image](markdown/redoc.png)
