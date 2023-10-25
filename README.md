##  EX03-Implementation-of-Linear-Regression-Using-Gradient-Descent

## AIM:
To write a program to predict the profit of a city using the linear regression model with gradient descent.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm:


1. Start the program.
2. import numpy as np.
3. Give the header to the data.
4. Find the profit of population.
5.Plot the required graph for both for Gradient Descent Graph and Prediction Graph.
6.End the program.


## Program:
```
/*
Program to implement the linear regression using gradient descent.
Developed by: Pavithra R
RegisterNumber:  212222230106

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

data=pd.read_csv("ex1.txt",header=None)
data

plt.scatter(data[0],data[1])
plt.xticks(np.arange(5,30,step=5))
plt.yticks(np.arange(-5,30,step=5))
plt.xlabel("Population of city (10,000s)")
plt.ylabel("Profit ($10,000")
plt.title("Profit Predication")

def computeCost(x,y,theta):
  m=len(y)
  h=x.dot(theta)
  square_err=(h-y)**2
  return 1/(2*m)*np.sum(square_err)
  
data_n=data.values
m=data_n[:,0].size
x=np.append(np.ones((m,1)),data_n[:,0].reshape(m,1),axis=1)
y=data_n[:,1].reshape(m,1)
theta=np.zeros((2,1))
computeCost(x,y,theta)

def gradientDescent(x,y,theta,alpha,num_iters):
  m=len(y)
  J_history=[]
  for i in range(num_iters):
    predictions=x.dot(theta)
    error=np.dot(x.transpose(),(predictions-y))
    descent=alpha*1/m*error
    theta-=descent
    J_history.append(computeCost(x,y,theta))
  return theta,J_history

theta,J_history = gradientDescent(x,y,theta,0.01,1500)
print("h(x) ="+str(round(theta[0,0],2))+" + "+str(round(theta[1,0],2))+"x1")

plt.plot(J_history)
plt.xlabel("Iteration")
plt.ylabel("$J(\Theta)$")
plt.title("Cost function using Gradient Descent")

plt.scatter(data[0],data[1])
x_value=[x for x in range(25)]
y_value=[y*theta[1]+theta[0] for y in x_value]
plt.plot(x_value,y_value,color="r")
plt.xticks(np.arange(5,30,step=5))
plt.yticks(np.arange(-5,30,step=5))
plt.xlabel("Population of City (10,000s)")
plt.ylabel("Profit ($10,000s")
plt.title("Profit Prediction")

def predict(x,theta):
    predictions=np.dot(theta.transpose(),x)
    return predictions[0]

predict1=predict(np.array([1,3.5]),theta)*10000
print("For population = 35,000, we predict a profit of $"+str(round(predict1,0)))

predict2=predict(np.array([1,7]),theta)*10000
print("For population = 70,000, we predict a profit of $"+str(round(predict2,0)))
*/
```

## Output:


DATASET:

![1](https://github.com/Pavithraramasaamy/Implementation-of-Linear-Regression-Using-Gradient-Descent/assets/118596964/06099a9a-27fb-423c-9623-9e98b88c8f2b)

## 1. Profit prediction graph 

![2](https://github.com/Pavithraramasaamy/Implementation-of-Linear-Regression-Using-Gradient-Descent/assets/118596964/1cee0eb9-c7ea-4194-957f-75c972e0c590)

## 2. Compute cost value 

![3](https://github.com/Pavithraramasaamy/Implementation-of-Linear-Regression-Using-Gradient-Descent/assets/118596964/db9d7b4f-b3af-48b7-98c3-0b26f43e500a)

## 3. h(x) value 


![4](https://github.com/Pavithraramasaamy/Implementation-of-Linear-Regression-Using-Gradient-Descent/assets/118596964/8ee38a81-b265-474a-ac0d-f669fddc1b69)


## 4.Cost function using Gradient Descent:

![5](https://github.com/Pavithraramasaamy/Implementation-of-Linear-Regression-Using-Gradient-Descent/assets/118596964/62b3d147-b378-4772-9f32-a5abd8c43e96)

## 5.Profit Prediction:

![6](https://github.com/Pavithraramasaamy/Implementation-of-Linear-Regression-Using-Gradient-Descent/assets/118596964/771a2858-79a6-430c-a1e8-d12b11763176)


## 6. Profit for the population of 35000


![7](https://github.com/Pavithraramasaamy/Implementation-of-Linear-Regression-Using-Gradient-Descent/assets/118596964/f658156b-ba64-4232-bf2c-0654dd47c9e8)

## 7. Profit for the population of 70000

![8](https://github.com/Pavithraramasaamy/Implementation-of-Linear-Regression-Using-Gradient-Descent/assets/118596964/c1c43a75-869b-421c-af72-f6b7e963ac93)





## Result:
Thus the program to implement the linear regression using gradient descent is written and verified using python programming.
