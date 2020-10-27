# TF-Recommenders with Kubernetes

## Requirements
* [Tensorflow Recommenders](https://github.com/tensorflow/recommenders)
* [Annoy](https://github.com/spotify/annoy)
* [Kubernetes](https://kubernetes.io/)
* [Docker](https://www.docker.com/)
* [Minikube](https://minikube.sigs.k8s.io/docs/start/)
* [Kubectl](https://kubernetes.io/docs/tasks/tools/install-kubectl/)

## Setup

### Getting Model assets

First, train the [multitask demo]() from the [TF-Recommenders repo](https://github.com/tensorflow/recommenders). 

Then save the user model as a ```SavedModel```:

```python
model.user_model.save("user_model")
```

Create an [Annoy index](https://github.com/spotify/annoy) of the ```model.movie_model()``` embeddings.

```python
import pickle
from annoy import AnnoyIndex

top_N = 100
embedding_dimension = 32

content_index = AnnoyIndex(embedding_dimension, "dot")
movie_embeddings = movies.enumerate().map(lambda idx, title: (idx, title, model.movie_model(title)))
content_index_to_movie = dict((idx, title) for idx, title, _ in movie_embeddings.as_numpy_iterator())

# We unbatch the dataset because Annoy accepts only scalar (id, embedding) pairs.
for movie_id, _, movie_embedding in movie_embeddings.as_numpy_iterator():
    index.add_item(movie_id, movie_embedding)

# Build a 10-tree ANN index.
index.build(10)
index.save("content_embedding.tree")

# save the index dictionary as well:

with open("content_index_to_movie.p", "wb") as fp:
    pickle.dump(content_index_to_movie, fp, protocol=pickle.HIGHEST_PROTOCOL)
```

Download the ```SavedModel``` directory, ```content_embedding.tree``` annoy index, and ```content_index_to_movie.p``` dictionary. Move the saved model, here named ```user_model/``` into the user-model directory in this repo. Move the ```content_embedding.tree``` and ```content_index_to_movie.p``` files into the ```recommender_app/app/``` directory in this repo.

```bash
$ mv path/to/user_model ./user-model/
$ mv path/to/content_embedding.tree ./recommender_app/app/
$ mv path/to/content_index_to_movie.p ./recommender_app/app/
```

### Setup minikube
The next steps assume you have Docker, minikube, and kubectl installed in your deployment environment. 

Start minikube cluster:
```bash
$ minikube start
$ eval $(minikube docker-env)   # use the Docker daemon inside the minikube cluster
```

Build the docker images for the app and tfserver:
```bash
$ cd ./recommender-app
$ docker build -f Dockerfile -t recommender-app:latest .

$ cd ../user-model/
$ docker build -f Dockerfile -t user-model:latest .
```

You should be able to see the docker images in minikube docker environment.
```bash
$ docker image ls

REPOSITORY                                TAG                 IMAGE ID            CREATED             SIZE
recommender-app                           latest              c91ca30f0059        5 minutes ago       2.35GB
user_model                                latest              57c44fd2fdd0        5 minutes ago       372MB
python                                    3.7                 805aebdf2363        6 days ago          876MB
k8s.gcr.io/kube-proxy                     v1.19.2             d373dd5a8593        4 weeks ago         118MB
k8s.gcr.io/kube-apiserver                 v1.19.2             607331163122        4 weeks ago         119MB
k8s.gcr.io/kube-controller-manager        v1.19.2             8603821e1a7a        4 weeks ago         111MB
k8s.gcr.io/kube-scheduler                 v1.19.2             2f32d66b884f        4 weeks ago         45.7MB
gcr.io/k8s-minikube/storage-provisioner   v3                  bad58561c4be        6 weeks ago         29.7MB
k8s.gcr.io/etcd                           3.4.13-0            0369cf4303ff        7 weeks ago         253MB
tensorflow/serving                        latest              e0fe79fbb64f        2 months ago        286MB
kubernetesui/dashboard                    v2.0.3              503bc4b7440b        3 months ago        225MB
k8s.gcr.io/coredns                        1.7.0               bfe3a36ebd25        4 months ago        45.2MB
kubernetesui/metrics-scraper              v1.0.4              86262685d9ab        6 months ago        36.9MB
k8s.gcr.io/pause                          3.2                 80d28bedfe5d        8 months ago        683
```

### Deploy pods

Navigate to the root of this repo where the ```.yaml``` files are and deploy both pods:

```bash
$ kubectl apply -f recommender-app.yaml
$ kubectl apply -f user-model.yaml
```
Expose the recommender-service external ip.

```bash
$ minikube service recommender-service

|-----------|---------------------|-------------|---------------------------|
| NAMESPACE |        NAME         | TARGET PORT |            URL            |
|-----------|---------------------|-------------|---------------------------|
| default   | recommender-service |        6000 | http://192.168.49.2:30242 |
|-----------|---------------------|-------------|---------------------------|
```

Then use the resulting ip to curl the server with a user id.
```bash
$ curl 192.168.49.2:30242/recommend/1  # get recommendations for user 1
```

And the server will return something like this:

```bash
{
    "user_id": [
        "1"
    ],
    "recommendations": [
        [
            'Toy Story (1995)',
            'Jumanji (1995)',
            ...
        ]
    ]
}
```
### Clean up
Clean up with the following commands:

```bash
$ kubectl delete services recommender-service user-model-service
$ kubectl delete deployments recommender-app user-model
$ minikube stop
$ minikube delete
```

