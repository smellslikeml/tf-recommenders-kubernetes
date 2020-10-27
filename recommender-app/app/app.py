import grpc
import pickle
import requests
import numpy as np
import tensorflow as tf
from annoy import AnnoyIndex

from tensorflow_serving.apis import predict_pb2
from tensorflow_serving.apis import prediction_service_pb2_grpc

from flask import Flask
from flask_restful import Api, Resource

top_N = 100
embedding_dimension = 32

# Load annoy index
content_index = AnnoyIndex(embedding_dimension, "dot")
content_index.load('content_embedding.tree')

# load index to content_id mapping
with open('content_index_to_movie.p', 'rb') as fp:
    content_index_to_movie = pickle.load(fp)

tf.compat.v1.app.flags.DEFINE_string('server', 'user-model-service.default.svc.cluster.local:8500',
        'PredictionService host:port')
FLAGS = tf.compat.v1.app.flags.FLAGS

app = Flask(__name__)
api = Api(app)

def get_user_embedding(user_id):
    channel = grpc.insecure_channel(FLAGS.server)
    stub = prediction_service_pb2_grpc.PredictionServiceStub(channel)
    # Send request
    # See prediction_service.proto for gRPC request/response details.
    request = predict_pb2.PredictRequest()
    request.model_spec.name = 'user_model'
    request.model_spec.signature_name = 'serving_default'
    request.inputs['string_lookup_1_input'].CopyFrom(
            tf.make_tensor_proto(user_id, shape=[1]))
    result = stub.Predict(request, 10.0)  # 10 secs timeout
    embedding = np.array(result.outputs["embedding_1"].float_val)
    return tf.convert_to_tensor(embedding)

class Recommender(Resource):
    def get(self, user_id):
        user_recs = {"user_id": [], "recommendations": []}
        user = tf.convert_to_tensor(user_id, dtype="string")
        query_embedding = get_user_embedding(user)
        candidates = content_index.get_nns_by_vector(query_embedding, top_N)
        candidates = [int(content_index_to_movie[x].decode("utf-8")) for x in candidates]
        user_recs["user_id"].append(user.numpy().decode("utf-8"))
        user_recs["recommendations"].append(candidates)
        return user_recs

api.add_resource(Recommender, '/recommend/<user_id>')

if __name__ == '__main__':
    app.run(debug=True, host="0.0.0.0")
