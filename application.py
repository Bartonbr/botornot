from flask import Flask, make_response, jsonify, request
import pandas as pd
import pickle

app = Flask(__name__)
model = pickle.load(open('models/model.pkl', 'rb'))


@app.route("/botornot", methods=['POST'])
def predict():

    data = request.get_json(force=True)

    data.update((x, [y]) for x, y in data.items())

    prediction = model.predict(pd.DataFrame.from_dict(data))

    return make_response(jsonify({'prediction': prediction[0]}), 200)


if __name__ == "__main__":
    app.debug = True
    app.run(host="0.0.0.0", port=8080)



