import pickle
import keras
import requests
from bs4 import BeautifulSoup
from keras.preprocessing.sequence import pad_sequences
from flask import Flask, request, jsonify
from flask_cors import CORS, cross_origin

cors = CORS(app)

app = Flask(__name__)

tokenizer = None
with open("newsfn-word-index.pickle", "rb") as handle:
    tokenizer = pickle.load(handle)

model = keras.models.load_model("newsfn.h5")
word_index = tokenizer.word_index


@app.route("/predict", methods=["POST"])
@cross_origin()
def predict():
    content = request.json
    url = content["url"]
    text = fetch_text(url)
    inp = [word_index[word] for word in text.split() if word in word_index]
    p = pad_sequences([inp], maxlen=500, padding="post", truncating="post")
    score = model.predict(p)[0][0]
    data = {"score": str(score)}
    headers = {"Content-Type": "application/json"}
    return jsonify(data), headers


def fetch_text(url):
    page = requests.get(url)
    soup = BeautifulSoup(page.content, "html.parser")
    c = soup.find("div", id=lambda x: x and x.startswith("content-body-"))
    ch = c.findChildren("p", recursive=False, text=True)
    text = ""
    for child in ch:
        text += child.getText()
    return text


if __name__ == "__main__":
    app.run(threaded=True, port=5000)

