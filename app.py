from flask import flask, render_template,request
import joblib
app = flask(__name__)

model = joblib.load("iris_model.pkl")

@app.route("/", methods=["GET","POST"])
def home():
    prediction = None
    flower_image = None

    if request.method == "POST":

        sepal_length = float(request.form["sepal_length"])
        sepal_width = float(request.form["sepal_width"])
        petal_length = float(request.form["petal_length"])
        petal_width = float(request.form["petal_width"])

        input_data = [
            sepal_length,
            sepal_length,
            petal_length,
            petal_length
        ]

        result = model.predict
        (input_data
         
         prediction = int(result[0]))

return render_template(
    "index.html",
    prediction=prediction
)