<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <title>Machine Learning as Service</title>

</head>
<body>
<form action = "http://localhost:5000/predict" method = "POST">
         <p>Sepal length <input type = "text" name = "sepal_length" /></p>
         <p>Sepal width <input type = "text" name = "sepal_width" /></p>
         <p>Petal length <input type = "text" name = "petal_length" /></p>
         <p>Petal width <input type ="text" name = "petal_width" /></p>
         <p><input type = "submit" value = "Predict Type" /></p>
      </form>
</body>
</html>



<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <title>Predicted Result</title>


</head>
<body>

 <h1>Predicted class type  {{ result }} </h1>

</body>
</html>


from flask import Flask, request, jsonify, render_template
from sklearn.externals import joblib
from ml.prediction import iris_model_load

app = Flask(__name__)

HTTP_BAD_REQUEST = 400


@app.route('/train', methods=['POST'])
def train():
    return "training"


@app.route('/')
def ml_as_servie_entry():
   return render_template('mlmainpage.html')

@app.route('/result',methods = ['POST', 'GET'])
def result():
   if request.method == 'POST':
      result = request.form
      return render_template("result.html",result = result)


@app.route('/predict', methods=['POST'])
def predict():
    form_data = request.form
    sepal_length = float(form_data["sepal_length"])
    sepal_width = float(form_data["sepal_width"])
    petal_length = float(form_data["petal_length"])
    petal_width = float(form_data["petal_width"])

    # Reject request that have bad or missing values.
    if (sepal_length is None or sepal_width is None
        or petal_length is None or petal_width is None):
        # Provide the caller with feedback on why the record is unscorable.
        message = ('Record cannot be scored because of '
                   'missing or unacceptable values. '
                   'All values must be present and of type float.')
        response = jsonify(status='error',
                           error_message=message)
        # Sets the status code to 400
        response.status_code = HTTP_BAD_REQUEST
        return response

    features = [[sepal_length, sepal_width, petal_length, petal_width]]

    try:
        label = iris_model_load.load_model(features)
    except Exception as err:
        message = ('Failed to score the model. Exception: {}'.format(err))
        response = jsonify(status='error', error_message=message)
        response.status_code = HTTP_BAD_REQUEST
        return response

    label = iris_model_load.load_model(features)
    return render_template("result.html", result=label)


if __name__ == '__main__':
    app.run(debug=True)

