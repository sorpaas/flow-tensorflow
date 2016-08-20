from tensorflow.examples.tutorials.mnist import input_data
import tensorflow as tf
mnist = input_data.read_data_sets('MNIST_data', one_hot=True)

## TODO Seperate mnist.train.images and mnist.train.labels
## TODO A way for figuring out shapes for different variables.

def softmax(i):
    W = tf.Variable(tf.zeros([784, 10]))
    b = tf.Variable(tf.zeros([10]))
    return tf.nn.softmax(tf.matmul(i, W) + b)

primitives = {
    "tensorflow/softmax": {
        "inports": ["in"],
        "fn": lambda i: softmax(i)
    },
    "tensorflow/log": {
        "inports": ["in"],
        "fn": lambda i: tf.log(i)
    },
    "tensorflow/msum": {
        "inports": ["in"],
        "fn": lambda i: -tf.reduce_sum(i, reduction_indices=[1])
    },
    "tensorflow/time": {
        "inports": ["a", "b"],
        "fn": lambda a, b: a * b
    },
    "tensorflow/mean": {
        "inports": ["in"],
        "fn": lambda i: tf.reduce_mean(i)
    }
}

def process_outport(graph, relevants, node_id, port, predefined):
    node = next(n for n in graph["nodes"] if n["id"] == node_id)
    type_graph = next(g for g in relevants if g["id"] == node["type"])

    assert primitives[type_graph["name"]] is not None
    params = map(lambda x: process_inport(graph, relevants, node_id, x, predefined), primitives[type_graph["name"]]["inports"])
    result = primitives[type_graph["name"]]["fn"](*params)
    return result

def process_inport(graph, relevants, node_id, port, predefined):
    node = next(n for n in graph["nodes"] if n["id"] == node_id)
    bounded = next((p for p in graph["inports"] if p["bound"]["node"] == node_id and p["bound"]["inport"] == port), None)
    connection = next((c for c in graph["connections"] if c["to"]["node"] == node_id and c["to"]["inport"] == port), None)

    if bounded is not None:
        assert predefined[bounded["name"]] is not None
        return predefined[bounded["name"]]
    elif connection is not None:
        return process_outport(graph, relevants, connection["from"]["node"], connection["from"]["outport"], predefined)

def train(trainee, trainee_relevants, minimize, minimize_relevants, data):
    assert data == "preset:mnist"
    data = mnist.train
    sess = tf.InteractiveSession()

    x = tf.placeholder(tf.float32, shape=[None, len(data.images[0])])

    y_outport = next(p for p in trainee["outports"] if p["name"] == "y")
    y = process_outport(trainee, trainee_relevants, y_outport["bound"]["node"], y_outport["bound"]["outport"], {"x": x})

    y_ = tf.placeholder(tf.float32, shape=[None, len(data.labels[0])])

    m_outport = next(p for p in minimize["outports"] if p["name"] == "out")
    m = process_outport(minimize, minimize_relevants, m_outport["bound"]["node"], m_outport["bound"]["outport"], {"y": y, "y_": y_})

    train_step = tf.train.GradientDescentOptimizer(0.5).minimize(m)

    sess.run(tf.initialize_all_variables())

    for i in range(1000):
        batch = data.next_batch(50)
        train_step.run(feed_dict={x: batch[0], y_: batch[1]})

    correct_prediction = tf.equal(tf.argmax(y,1), tf.argmax(y_,1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    print(accuracy.eval(feed_dict={x: mnist.test.images, y_: mnist.test.labels}))

from flask import Flask, request, Response
import json
app = Flask(__name__)
app.config["DEBUG"] = True

@app.route("/train", methods=["POST"])
def http_train():
    assert request.headers["Content-Type"] == "application/json"

    train(request.json["trainee"]["target"], request.json["trainee"]["relevants"],
          request.json["minimize"]["target"], request.json["minimize"]["relevants"],
          request.json["data"])

    return Response(json.dumps({"status": "ok"}), status=200, mimetype="application/json")

if __name__ == "__main__":
    app.run()
