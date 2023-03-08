
## The k nearest neighbor in probability simplex (KNM-PS) 


## Notation: c = number of classes in the multi-class problem, n = size of training data, N = size of test data

## Input:
## y_mat_tr = n by c matrix in which each row contains binary elements (1 if the instance belongs to the class corresponding to the column and 0 otherwise)
## pred_tr = n by c matrix in which each row contains the initial class probability estimates of the training data instances
## pred_ts = N by c matrix in which each row contains the initial class probability estimates of the test data instances

## Output: 
## pred_ts_knnps = N by c matrix in which each row contains the class probability estimates of the test data instances re-calibrated by KNN-PS

## Note:
## The function uses a function from the FNN package. Install and load the package before running the function

# install.package("FNN") 
library(FNN)

KNNPS <- function(y_mat_tr,pred_tr,pred_ts)
{
  # a set of k values from which the optimal k will be determined
  k_list <- seq(10,200,by=10) # other ranges can be used.
  
  # Object idx contains all indices of knn data points for all training instances
  idx <- knnx.index(pred_tr, pred_tr, k=max(k_list)+1)
  idx <- idx[,-1]
  
  # identifying k that minimizes the brier score on training data
  pred_k <- matrix(0,nrow=nrow(pred_tr),ncol=ncol(pred_tr))
  brier_vec <- numeric(0)
  for(j in 1:length(k_list))
  {
    for(i in 1:nrow(idx))
    {
      pred_k[i,] <- apply(y_mat_tr[idx[i,1:k_list[j]],],2,mean)  
    }
    brier_vec[j] <- mean(apply((y_mat_tr-pred_k)^2,1,sum))
  }
  k <- k_list[which.min(brier_vec)]
  
  # KNN-PS using the optimal k
  idx <- knnx.index(pred_tr, pred_ts, k=k)
  pred_ts_knnps <- matrix(0,nrow=nrow(pred_ts),ncol=ncol(pred_ts))
  for(i in 1:nrow(idx))
  {
    pred_ts_knnps[i,] <- apply(y_mat_tr[idx[i,],],2,mean)  
  }
  
  # a list containing that (1) class probabilities re-calibrated by KNN-PS and (2) the optimal k
  list(pred_ts_knnps,k)
}
