## 📚 Academic Course Project
As part of an academic course, I implemented the K-means++ algorithm in C Python.


## 🚀 Practical Applications of K-means++ Algorithm
The K-means++ algorithm has many real-world uses. For example, in marketing, it helps companies group customers based on their buying habits for better targeting. In image processing, it compresses images by grouping similar pixels together, saving storage space without losing quality. Also, in social networks, it identifies communities by grouping users with similar interactions, useful for targeted content and network analysis.

## 🛠️ How to Use the Project
#### 1. Provide arguments 
* K - the number of required clusters
* inter - maximum iteration count
* eps - convergence value
#### 2. Data Preparation 
Combine input files by inner join using the first column as a key and sort the data points in ascending order.
#### 3. Interfacing with C Extension 
Import the C module `import mykmeanssp`, call the `fit()` method with **initial centroids** and **data points**, and **retrieve the final centroids**.

<br><br><br>

### 📝 Appendix: Special Mathematical Matrices
The implementation involves several mathematical concepts and algorithms, including:

#### Euclidean Distance: Measure of distance between data points.
#### Cluster Assignment: Assigning data points to the closest cluster based on distance.
#### Update Centroids: Recalculating centroids based on data points in each cluster.
#### Convergence Criteria: Checking for convergence based on centroid updates and maximum iteration count.
