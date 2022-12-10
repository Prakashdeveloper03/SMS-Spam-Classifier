from joblib import load
from fastapi import FastAPI, Form, Request
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

app = FastAPI()  # instance of FastAPI class

# mount static folder files to /static route
app.mount("/static", StaticFiles(directory="static"), name="static")

# loads the ML models
classifier = load(open("models/spamClassifier.pkl", "rb"))
transform = load(open("models/transform.pkl", "rb"))

# sets the templates folder for the app
templates = Jinja2Templates(directory="templates")


@app.get("/", response_class=HTMLResponse)
async def home(request: Request):
    """
    Function to render `base.html` at route '/' as a get request
    __Args__:
    - __request (Request)__: request in path operation that will return a template
    __Returns__:
    - __TemplateResponse__: render `base.html`
    """
    return templates.TemplateResponse("base.html", {"request": request})


@app.post("/predict", response_class=HTMLResponse)
async def predict(
    request: Request,
    message: str = Form(...),
):
    """
    Function to predict spam status and shows the result by rendering `base.html` at route '/predict'
    __Args__:
    - __request (Request)__: request in path operation that will return a template
    - __message (str)__: message text
    Returns:
    - __Template Response__: render `base.html`
    """
    data = [message]  # str -> [str]
    vect = transform.transform(
        data
    ).toarray()  # convert a collection of text documents to a matrix of token counts
    prediction = classifier.predict(
        vect
    )  # prediction using multinomial naive bayes model
    print(prediction)
    output = (
        "Gotcha! This is a SPAM message."
        if prediction == 1
        else "Great! This is NOT a spam message."
    )
    return templates.TemplateResponse(
        "base.html", context={"request": request, "prediction": output}
    )
