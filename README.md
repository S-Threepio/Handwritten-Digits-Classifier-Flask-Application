# ML application for MNIST classification
- Created a model by training the MNIST data on a custom Convolutional Neural Network.
- Utilised the Flask server to accept POST requests with images which gives back the identified digit.

# Docker build
- Utilized docker to create a docker image using dockerfile
- Optimized the space using Tensor CPU capabilities instead of CUDA reducing the image size by 75%
- Pushed the image to Dockerhub

# AWS EKS
- Deployed the application inside a kubernetes cluster using AWS EKS
- Created master nodes with EC2
- Utilized eksctl for seamless deployment

# GCP GKE
- Deployed the application inside a kubernetes cluster using GCP GKE
- Utilized the artifact registry to push the docker image
- Used the GCP monitoring API to monitor clusters

Utilzed the YAML files for deployment and creation of service for kubernetes clusters
