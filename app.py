import numpy as np
import pandas as pd
from flask import Flask, request, render_template
import tensorflow as tf

app = Flask(__name__)

# Load your trained model
model = tf.keras.models.load_model("crime_prediction_model.h5")

# Load the correct dummy columns used during training (one-hot encoded structure)
dummy_cols = pd.read_csv("dummy_columns.csv", header=None)
expected_columns = dummy_cols[0].tolist()

@app.route("/", methods=["GET", "POST"])
def index():
    prediction = None

    if request.method == "POST":
        try:
            # Get form inputs
            year = int(request.form["year"])
            state = request.form["state_ut"]
            district = request.form["district"]

            # Create dataframe from user input
            input_dict = {
                'year': year,
                'state_ut_' + state: 1,
                'district_' + district: 1
            }

            # Create a DataFrame with one row
            input_df = pd.DataFrame([input_dict])

            # Add any missing columns as 0
            for col in expected_columns:
                if col not in input_df.columns:
                    input_df[col] = 0

            # Ensure column order matches training
            input_df = input_df[expected_columns]

            # Predict
            input_array = input_df.values.astype(np.float32)
            prediction = model.predict(input_array)[0][0]
            prediction = round(prediction)

        except Exception as e:
            prediction = f"Error: {str(e)}"

    return render_template("index.html", prediction=prediction)

if __name__ == "__main__":
    app.run(debug=True)
