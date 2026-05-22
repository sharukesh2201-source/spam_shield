from flask import Flask, request, jsonify
import pickle

app = Flask(__name__)

model = pickle.load(open("spam_model.pkl", "rb"))
vectorizer = pickle.load(open("vectorizer.pkl", "rb"))

@app.route('/check', methods=['POST'])
def check():
    data = request.json
    message = data['message']

    vector = vectorizer.transform([message])
    result = model.predict(vector)[0]

    return jsonify({"result": result})

app.run(host="0.0.0.0", port=5000)