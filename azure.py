import azure.functions as func
import pickle

def main(req: func.HttpRequest) -> func.HttpResponse:
    with open('iris_model.pkl', 'rb') as f:
        model = pickle.load(f)

    req_body = req.get_json()
    sepal_length = req_body.get('sepal_length')
    sepal_width = req_body.get('sepal_width')
    petal_length = req_body.get('petal_length')
    petal_width = req_body.get('petal_width')

    prediction = model.predict([[sepal_length, sepal_width, petal_length, petal_width]])
    return func.HttpResponse(str(prediction[0]))

