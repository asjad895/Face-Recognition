# Face Recognition 
<b><span style="color:green">Using Siamese Network for Face Recognition in One Shot Learning.</span></b>

Welcome to the README file for our project on **face recognition for login** using the **Siamese network** in one-shot learning method. In today's world, security is a critical aspect of every system, and face recognition is a reliable technology to achieve it. With the increasing demand for a seamless login experience, face recognition is becoming more popular as an authentication method.

Our project focuses on using the Siamese network for face recognition, which is a type of **neural network** that learns to recognize faces in a **one-shot learning** method. The Siamese network is a powerful tool that can learn to recognize faces even when given very few examples. This approach is particularly useful in scenarios where only one image of a person's face is available for authentication.

In this README file, we will explain the methodology and implementation of our face recognition project using the Siamese network in one-shot learning. We will also provide the necessary instructions for running our code and reproducing our results. So, let's get started!

## our adopted method.
**Identification via Verification**
  * Identification requires training a classifier to assign input face images to specific identities in the database.
  * This approach becomes inefficient and non-scalable as the number of people in the database increases.
  * Adding a new person to the database requires re-training the identification system from scratch with an increased number of neurons.
  * The more effective approach is using the similarity-based comparison approach in verification.
  * In verification, the algorithm is run multiple times for each of the K Face IDs in the database to find the match for the input image.
  * The identity of the input face is determined by the Face ID for which the binary output of the verification algorithm is true.
  * The benefit of using verification for identification is that adding a new face to the database only requires running the verification algorithm K+1 times without retraining the network.
  * The classifier-based identification approach inhibits scalability due to the fixed number of neurons in the final layer.
  * On the other hand, the similarity-based comparison approach used in verification is more efficient and scalable when new faces are added to the face database.


Figure:Identification via Verification 
#adding figure link

The benefit of using verification for identification is that if we add a new face to our database of faces, we just need to run the verification algorithmK+1times for identification without the need to re-train the network.

Thus we observe that the classifier-based identification approach inhibits scalability due to the fixed number of neurons in the final layer. However, the similarity-based comparison approach used in verification is more efficient and scalable when new faces are added to the face database.

**Learning Metric.**
 * metric learning is a paradigm where our network is trained in such a way that representations of similar images are close to each other in an embedding space and images of different objects are farther apart. The idea of “similarity” here is defined based on a specific distance metric in the embedding space, which quantifies semantic similarity and whether the images belong to the same or different object or person.  
 **Triplet Loss**
  * The triplet loss is a slightly enhanced version of the contrastive loss that we will use for building our face recognition application. The triplet loss function shares the same basic principles and characteristics as the pairwise contrastive loss function However. its formulation is based on a triplet data sample, which is slightly different from the pairwise loss discussed previously.    
  
The triplet loss function requires a sample consisting of a triplet of images, namely Anchor, Positive, and Negative.
 The **anchor** and **positive** images belong to the same class or person, while the **negative** image belongs to a different class or person than the positive image. Moreover, the anchor and positive images are different instances of the same person, depicting them in different looks, varied poses, hairstyles, backgrounds, etc.   
  
 **Formula**
 L(A, P, N) = max(||f(A) - f(P)||² - ||f(A) - f(N)||² + margin, 0)  
  
where:

A is the anchor image
P is the positive image
N is the negative image
f(.) is the embedding function that maps an input image to a feature vector in a high-dimensional space,
||.||² denotes the Euclidean distance between two feature vectors,
margin is a hyperparameter that represents the minimum distance between the positive and negative feature vectors.
The triplet loss function aims to minimize the distance between the anchor and positive feature vectors while maximizing the distance between the anchor and negative feature vectors. This encourages the network to learn discriminative features that can differentiate between different faces, making it suitable for face recognition applications.  
***  
  
```python
import torch.nn.functional as F

# Define a function for computing the triplet loss
def triplet_loss(anchor, positive, negative, margin):
    # Compute the Euclidean distance between the anchor and positive samples
    distance_pos = F.pairwise_distance(anchor, positive, 2)
    
    # Compute the Euclidean distance between the anchor and negative samples
    distance_neg = F.pairwise_distance(anchor, negative, 2)
    
    # Compute the triplet loss as the mean of the ReLU value of the difference between distances and the margin
    loss = torch.mean(F.relu(distance_pos - distance_neg + margin))
    
    # Return the computed loss
    return loss

```