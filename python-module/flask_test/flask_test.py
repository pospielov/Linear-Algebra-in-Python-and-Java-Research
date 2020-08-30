from flask import Flask
from flask import Response
import numpy as np
app = Flask(__name__)


embeddings = np.genfromtxt("embeddings.txt", delimiter=',')
embeddings = embeddings[1:10001]


@app.route("/", methods=['POST'])
def endpoint():
    from flask import request
    body = request.get_json()
    embeddings1 = np.expand_dims(np.array(body), axis=0)

    return Response('{"result": "%s"}' % str(test(embeddings1)), mimetype="application/json")


def test(embeddings1):
    embeddings2 = embeddings

    embeddings1 = embeddings1/np.linalg.norm(embeddings1, axis=1, keepdims=True)
    embeddings2 = embeddings2/np.linalg.norm(embeddings2, axis=1, keepdims=True)
    diff = np.subtract(embeddings1, embeddings2)
    dist = np.sum(np.square(diff), 1)
    # print(np.argmax(dist));
    return np.max(dist)


if __name__ == "__main__":
    app.run(host='0.0.0.0')