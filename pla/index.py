import random
import time
# from matplotlib.pyplot import plot

#default functions
def sgn(x):
    if x>0:
        return 1
    elif x<0:
        return -1
    else:
        return 0

def f(x,y):
    return sgn(5*x+4*y-20)

grid=([1,1],[1,2],[3,4],[2,3],[4,1],[5,2],[6,3],[7,4],[8,5],[9,6])                                        
wt=[0,0]
bias=20

def g(x,y):
    return sgn(wt[0]*x+wt[1]*y+bias)

def perceptron(pos):
    ans=pos[0]*wt[0]+pos[1]*wt[1]
    return sgn(ans-bias)

LR=0.001
def training(): 
    i=0
    while True:
        misclassified=[]
        for data in grid:
            index=0
            if f(data[0],data[1])!=g(data[0],data[1]):
                misclassified.append(data)
                index+=1
        if not misclassified:
            print("Done!")
            break
        print(len(misclassified), " misclassified points")
        x1,y1=random.choice(misclassified)
        wt[0]+=f(x1,y1)*x1*LR
        wt[1]+=f(x1,y1)*y1*LR
        i+=1
        print("W0 : ",wt[0] , "  W1 : ",wt[1],"  iterations : ",i)

        print("")

training()
