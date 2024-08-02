# ND-Net++

Created by Carlos Tojal, Daniel Carias, Luis Conde Bento, Hugo Costelha and Catarina Reis from Polytechnic Institute of Leiria.

ND-Net++ is a PointNet++-based point cloud classification, part segmentation and semantic segmentation neural network. ND is an abbreviation for "Normal Distribution".

PointNet++ adopts multiple sampling and grouping stages to hierarchically extract features while abstracting the point cloud. To reduce the size of the network, we propose a new architecture sharing the same hierarchical concept, but estimating normal distributions instead of performing sampling and grouping.

The grouping stage of the PointNet++ generates $K$ groups with $N$ 3-dimensional neighbors ($B \times K \times N \times 3$ tensor). Instead, a normal distribution can represent a neighborhood of points by its mean 3-dimensional vector and its $3 \times 3$ covariance matrix only.
