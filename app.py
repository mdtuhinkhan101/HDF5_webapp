from flask import Flask, render_template, request
import numpy as np
import tensorflow as tf

app = Flask(__name__)

# ---------------- Load the model ----------------
model = tf.keras.models.load_model("model.h5")

# ---------------- Home page ----------------
@app.route("/")
def index():
    return render_template("index.html")

# ---------------- Prediction route ----------------
@app.route("/predict", methods=["POST"])
def predict():
    try:
        # Collect 8 inputs
        f1 = float(request.form["f1"])
        f2 = float(request.form["f2"])
        f3 = float(request.form["f3"])
        f4 = float(request.form["f4"])
        f5 = float(request.form["f5"])
        f6 = float(request.form["f6"])
        f7 = float(request.form["f7"])
        f8 = float(request.form["f8"])

        # Convert into numpy array
        data = np.array([[f1, f2, f3, f4, f5, f6, f7, f8]])

        # Predict
        prediction = model.predict(data)[0]

        # If classification model → show class
        result = np.argmax(prediction)

        return render_template("index.html", prediction=result)

    except Exception as e:
        return render_template("index.html", prediction=f"Error: {e}")

if __name__ == "__main__":
    app.run(debug=True)
