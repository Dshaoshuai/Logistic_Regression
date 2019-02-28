import numpy as np
import h5py
import matplotlib.pyplot as plt

def load_dataset():
    train_dataset=h5py.File('datasets/train_catvnoncat.h5',"r")
    train_set_x_orig=np.array(train_dataset["train_set_x"][:])#np.array()用于数组创建
    train_set_y_orig=np.array(train_dataset["train_set_y"][:])#train set labels

    test_dataset=h5py.File('datasets/test_catvnoncat.h5',"r")
    test_set_x_orig=np.array(test_dataset["test_set_x"][:])#your test set features [:]表示遍历列表
    test_set_y_orig=np.array(test_dataset["test_set_y"][:])#your test set labels

    classes=np.array(test_dataset["list_classes"][:])#the list of classes

    train_set_y_orig=train_set_y_orig.reshape((1,train_set_y_orig.shape[0]))
    test_set_y_orig=test_set_y_orig.reshape((1,test_set_y_orig.shape[0]))

    return train_set_x_orig,train_set_y_orig,test_set_x_orig,test_set_y_orig,classes
train_set_x_orig,train_set_y,test_set_x_orig,test_set_y,classes=load_dataset()

index=25
# plt.imshow(train_set_x_orig[index])
# plt.show()
# print("train_set_y="+str(train_set_y))
# print("classes:"+str(classes))

# print("y="+str(train_set_y[:,index])+", it's a "+classes[np.squeeze(train_set_y[:,index])].decode("utf-8")+"' picture")
#print("train_set_y: "+str(train_set_y[0]))

m_train=train_set_y.shape[1]
m_test=test_set_y.shape[1]
num_px=train_set_x_orig.shape[1]

# print("训练集的数量： m_train= "+str(m_train))
# print("测试集的数量： m_test= "+str(m_test))
# print("每张图片的宽/高： num_px= "+str(num_px))
# print("每张图片的大小：（"+str(num_px)+", "+str(num_px)+", 3")
# print("训练集_图片的维数： "+str(train_set_x_orig.shape))
# print("训练集_标签的维数："+str(train_set_y.shape))
# print("测试集_图片的维数："+str(test_set_x_orig.shape))
# print("测试集_标签的维数："+str(test_set_y.shape))


train_set_x_flatten=train_set_x_orig.reshape(train_set_x_orig.shape[0],-1).T
test_set_x_flatten=test_set_x_orig.reshape(test_set_x_orig.shape[0],-1).T
# print("训练集降维后的维度："+str(train_set_x_flatten.shape))
# print("训练集_标签的维度："+str(train_set_y.shape))
# print("测试集降维后的维度："+str(test_set_x_flatten.shape))
# print("测试集_标签的维度："+str(test_set_y.shape))

train_set_x=train_set_x_flatten/255
test_set_x=test_set_x_flatten/255

def sigmoid(z):
    s=1/(1+np.exp(-z))
    return s

# print("=============测试sigmoid=============")
# print("sigmoid(0)= "+str(sigmoid(0)))
# print("sigmoid(9.2)= "+str(sigmoid(9.2)))

#初始化参数w和b
def initialize_with_zeros(dim):
    w=np.zeros(shape=(dim,1))
    b=0
    assert(w.shape==(dim,1))
    assert(isinstance(b,float) or isinstance(b,int))
    return (w,b)

# print("实数的shape："+str((1).shape))

#进行一次批处理后得到的cost，dw，db，其用于本轮的梯度下降
def propagate(w,b,X,Y):
    m=X.shape[1]

    #正向传播
    A=sigmoid(np.dot(w.T,X)+b)
    cost=(-1/m)*np.sum(Y*np.log(A)+(1-Y)*(np.log(1-A)))

    #反向传播
    dw=(1/m)*np.dot(X,(A-Y).T)
    db=(1/m)*np.sum(A-Y)

    assert(dw.shape==w.shape)
    assert(db.dtype==float)
    cost=np.squeeze(cost)
    assert(cost.shape==())

    grads={
        "dw":dw,
        "db":db
    }
    return (grads,cost)

# print("=============测试propagate===============")
# w,b,X,Y=np.array([[1],[2]]),2,np.array([[1,2],[3,4]]),np.array([[1,0]])
# grads, cost=propagate(w,b,X,Y)
# print("dw= "+str(grads["dw"]))
# print("db= "+str(grads["db"]))
# print("cost= "+str(cost))

def optimize(w,b,X,Y,num_iterations,learning_rate,print_cost=False):
    costs=[]

    for i in range(num_iterations):
        grads,cost=propagate(w,b,X,Y)

        dw=grads["dw"]
        db=grads["db"]

        w=w-learning_rate*dw
        b=b-learning_rate*db

        if i%100 ==0:
            costs.append(cost)
        if (print_cost) and (i%100==0):
            print("迭代的次数：%i，误差值：%f" % (i,cost))

    params={
            "w":w,
            "b":b
    }
    grads={
            "dw":dw,
            "db":db
    }
    return (params,grads,costs)

def predict(w,b,X):
    m=X.shape[1]

    Y_prediction=np.zeros((1,m))
    w=w.reshape(X.shape[0],1)

    #计算猫在图片中出现的概率
    A=sigmoid(np.dot(w.T,X)+b)
    for i in range(A.shape[1]):
        #将概率a[0,i]转换为实际预测p[0,i]
        Y_prediction[0][i]=1 if A[0][i]>0.5 else 0    #二维列表用np.array()转numpy.array后可使用.shape， 此时[0,1]和[0][1]的意义相同
    assert(Y_prediction.shape==(1,m))
    return Y_prediction

def model(X_train,Y_train,X_test,Y_test,num_iterations=2000,learning_rate=0.5,print_cost=False):
        w,b=initialize_with_zeros(X_train.shape[0])    #初始化参数
        parameters, grads, costs=optimize(w,b,X_train,Y_train,num_iterations,learning_rate,print_cost)    #得到迭代num_iterations之后的参数，梯度和成本列表

        #从字典"参数"中检索参数w和b
        w,b=parameters['w'],parameters['b']

        #预测测试/训练集的例子
        Y_prediction_test=predict(w,b,X_test)        #采用正向传播得到测试集的预测结果
        Y_prediction_train=predict(w,b,X_train)

        #打印训练后的准确性
        print("训练集准确性：",format(100-np.mean(np.abs(Y_prediction_train-Y_train))*100),"%")
        print("测试准确性：",format(100-np.mean(np.abs(Y_prediction_test-Y_test))*100),"%")

        d={
            "costs":costs,
            "Y_prediction_test":Y_prediction_test,
            "Y_prediction_train":Y_prediction_train,
            "w":w,
            "b":b,
            "learning_rate":learning_rate,
            "num_iterations":num_iterations
        }

        return d

print("============测试model=============")
d=model(train_set_x,train_set_y,test_set_x,test_set_y,num_iterations=2000,learning_rate=0.005,print_cost=False)

costs=np.squeeze(d["costs"])
plt.plot(costs)
plt.ylabel("cost")
plt.xlabel("iterations (per hundreds)")
plt.title("Learning rate= "+str(d["learning_rate"]))
plt.show()

learning_rates=[0.01,0.001,0.0001]
models={}
for i in learning_rates:
    print("learning rate is: "+str(i))
    models[str(i)]=model(train_set_x,train_set_y,test_set_x,test_set_y,num_iterations=1500,learning_rate=i,print_cost=False)
    print("\n"+"-----------------------------------------------"+"\n")

for i in learning_rates:
    plt.plot(np.squeeze(models[str(i)]["costs"]),label=str(models[str(i)]["learning_rate"]))

plt.ylabel("cost")
plt.xlabel("iterations (per hundreds)")

legend=plt.legend(loc="upper center",shadow=True)
frame=legend.get_frame()
frame.set_facecolor("0.90")
plt.show()

# a=np.array([[1,2],[3,4]])
#
#
# print(a[0,1])