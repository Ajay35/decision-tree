
# coding: utf-8

# # API of using the decision tree classifier is as following:
### reading the data
df = pd.read_csv("train.csv")
# dividing data in 80-20 in train and validate datasets
train_data, validate_data = train_test_split(df, test_size=0.2)
## Decision tree algorithm on train datasets
tree = decision_tree_algorithm(train_df)
## classifying sample tests
predict(test_data,tree)
# finding accuracy of test by decision tree prediction
accuracy = calculate_accuracy(test_df, tree)
# ## requied import 

# In[700]:


import numpy as np
import pandas as pd
import scipy as sc
import matplotlib.pyplot as plt
import seaborn as sns
import random
from sklearn.metrics import precision_recall_fscore_support as get_all_scores
import pprint


# ## load data

# In[701]:


df=pd.read_csv('train.csv')
df.head()


# In[702]:


df=pd.read_csv('train.csv')
#df = pd.read_csv("train.csv",usecols=['Work_accident','promotion_last_5years','department','salary','left'])
df.head()


# 
# ## Dividing 80%-20% of data for train and validation purpose

# In[703]:


def divide(df,validation_data_size):
    indices = df.index.tolist()
    validation_data_size=round(validation_data_size*len(df))
    validation_indices = random.sample(population=indices, k=validation_data_size)
    train_data = df.drop(validation_indices)
    validation_data = df.loc[validation_indices]
    return train_data,validation_data


# In[704]:


random.seed(0)
train_data,validation_data = divide(df,validation_data_size=0.2)


# In[705]:


data = train_data.values
data[:10]


# # *******Helper functions***********

# ### Check if data has only one unique class

# In[706]:


def is_unique_class(data):
    label=data[:, -1]
    classes=np.unique(label)
    if len(classes)==1:
        return True
    else:
        return False


# ### Classify which has class has majority

# In[707]:


def get_class(data):
    #fetch target class labels
    label=data[:,-1]
    #get unique classes and no of unique class
    classes,no_of_classes= np.unique(label,return_counts=True)
    classification=classes[no_of_classes.argmax()]
    return classification


# ### Finding possible divisions for column

# In[1]:


def possible_division(data):
    
    divisions = {}
    _, n_columns = data.shape
    for column in range(n_columns - 1):
        values=data[:, column]
        unique_values=np.unique(values)
        type_of_feature = types[column]
        
        #for continuous data check > and < conditions
        if type_of_feature == "continuous":
            divisions[column] = []
            for index in range(len(unique_values)):
                if index != 0:
                    cur=unique_values[index]
                    prev=unique_values[index-1]
                    division = (cur+ prev)/2
                    divisions[column].append(division)
        
        # for categorical feature check == and != conditions
        else:
            divisions[column]=unique_values
    
    return divisions


# ### divide data

# In[683]:


def divide_data(data,column,value):
    
    column_values = data[:,column]

    type_of_feature = types[column]
    if type_of_feature == "continuous":
        data_on_left = data[column_values <=value]
        data_on_right = data[column_values >value]
    
    # feature is categorical   
    else:
        data_on_left = data[column_values==value]
        data_on_right = data[column_values!=value]
    return data_on_left, data_on_right


# In[709]:


def get_best_division(data, divisions):
    
    min_entropy = 500
    for column in divisions:
        for value in divisions[column]:
            data_on_left, data_on_right = divide_data(data,column=column,value=value)
            current_entropy = get_total_entropy(data_on_left, data_on_right)
            if current_entropy <= min_entropy:
                min_entropy = current_entropy
                division_column=column
                division_value=value
    
    return division_column,division_value


# ### calculate entropy,gini index to select the feature with lowest total entropy or gini index

# In[684]:


def entropy(data):
    label=data[:, -1]
    _,counts=np.unique(label,return_counts=True)
    probabilities=counts/counts.sum()
    ent=sum(probabilities * -np.log2(probabilities))
    #print(type(probabilities))
    #gini=1-sum([ i**2 for i in probabilities]) 
    #return gini
    return ent
    #return 1-np.max(probabilities)




# In[685]:


def get_total_entropy(data_on_left, data_on_right):
    p_data_on_left = len(data_on_left) / (len(data_on_left) + len(data_on_right))
    p_data_on_right = len(data_on_right) / (len(data_on_left) + len(data_on_right))
    total_entropy=(p_data_on_left*entropy(data_on_left)+p_data_on_right*entropy(data_on_right))
    return total_entropy


#  ### Get type of feature (categorical or continuous)

# In[710]:


def get_feature_type(data):
    feat_types=[]
    threshold = 15
    for feature in data.columns:
        if feature != "label":
            unique_vals=data[feature].unique()
            example_value=unique_vals[0]
            # for categorical data entries might be string and unique values may be low than threshold
            if (isinstance(example_value, str)) or (len(unique_vals) <= threshold):
                feat_types.append("categorical")
            else:
                feat_types.append("continuous")
    
    return feat_types


# ### Decision tree Algorithm

# In[2]:


no_of_nodes=0
def increment(c):
    global no_of_nodes
    no_of_nodes += c
def set_zero():
    global no_of_nodes
    no_of_nodes=0
def get_nodes():
    global no_of_nodes
    return no_of_nodes

def id3_tree(df, cnt=0, min_samples=2, max_depth=8):
    #for first function call categorize columns into continuous and categorical
    if cnt == 0:
        global heading, types
        types = get_feature_type(df)
        data = df.values
        heading = df.columns
    else:
        data = df           
    if len(data)==0 :
        return
    
    # base case
    #if unique instance of yes or no in data
    # length of tree reached maximum
    #data has features than min_samples 
    if (is_unique_class(data)) or (len(data) < min_samples) or (cnt == max_depth):
        classification = get_class(data)
        increment(1)
        return classification

    # still data is not pure according to provided parameters
    else:    
        cnt += 1
        increment(2)
        # helper functions 
        possible_divisions = possible_division(data)
        column,value = get_best_division(data, possible_divisions)
        data_on_left,data_on_right=divide_data(data,column,value)
        
        # extract feature at current node
        feature_name = heading[column]
        if types[column]=="continuous":
            node = "{} <= {}".format(feature_name,value)
            
        # feature is categorical
        else:
            node = "{} = {}".format(feature_name,value)
        
        # new sub-tree
        sub_tree = {node: []}
        
        # find answers (recursion)
        pos=id3_tree(data_on_left, cnt, min_samples, max_depth)
        neg=id3_tree(data_on_right,cnt, min_samples, max_depth)
        
        # If the answers are the same, then there is no point in asking the qestion.
        # This could happen when the data is classified even though it is not pure
        # yet (min_samples or max_depth base case).
        if pos==neg:
            sub_tree=pos
        else:
            sub_tree[node].append(pos)
            sub_tree[node].append(neg)
        return sub_tree


# ### Building a tree

# In[713]:


tree = id3_tree(train_data, max_depth=10)
pprint.pprint(tree)
print("number of nodes",no_of_nodes) 


# ## Prediction

# In[714]:


sample_entry=validation_data.iloc[0]


# In[715]:


def predict(sample, tree):
    node = list(tree.keys())[0]
    feature_name, cop, value = node.split(" ")

    # compare the node value for == or <=
    if cop == "<=":
        if sample[feature_name] <= float(value):
            val = tree[node][0]
        else:
            val = tree[node][1]
    
    # for categorical feature
    else:
        if str(sample[feature_name]) == value:
            val=tree[node][0]
        else:
            val=tree[node][1]

    # base case
    if not isinstance(val, dict):
        return val
    
    # go deeper into the tree
    else:
        sub_tree=val
        return predict(sample,sub_tree)


# In[3]:


print(sample_entry)
predict(sample_entry, tree)


# ### Calculate Accuracy

# In[717]:


def get_accuracy(data, tree):

    data["prediction"] = data.apply(predict,args=(tree,),axis=1)
    data["correct_prediction"]= data["prediction"]==data["left"]
    acc= data["correct_prediction"].mean()
    #df["prediction"].head()
    a,b,c,d=get_all_scores(data["left"],data["prediction"])
    #calculation of precision ,recall and F1 score
    print("precision:",a[0])
    print("recall:",b[0])
    print("fscore:",c[0])
    return acc


# In[718]:


print(get_accuracy(validation_data,tree))


# # 4. Visualization of Training Data

# In[719]:


#plt.scatter(df.average_montly_hours, df.satisfaction_level)
dt=pd.read_csv('train.csv')
sns.scatterplot(x="satisfaction_level",y="last_evaluation",data=dt,hue='left')
#sns.lmplot(x="number_project",y="satisfaction_level",data=dt,hue='left',fit_reg=True)
#sns.pairplot(dt,hue='left')


# # 5. Number of nodes vs Validation error

# In[720]:


validation_error=[]
number_of_nods=[]
for i in range (5,20):
    global no_of_nodes
    #print("value of no of nodes:",no_of_nodes)
    set_zero()
    tree = id3_tree(train_data,max_depth=i)
    print("value of no of nodes:",no_of_nodes)
    accuracy =get_accuracy(validation_data,tree)
    print("test:",i-4)
    number_of_nods.append(get_nodes())
    validation_error.append(1-accuracy)
    accuracy
    



# # Graph :Number of nodes vs Validation Error

# In[721]:


plt.plot(number_of_nods,validation_error)
plt.xlabel("Number of Nodes")
plt.ylabel("Validation Error")
plt.show()


# 
# # 5.Number of Nodes vs Training Error
# 
# 

# In[722]:


#train_df.drop(['classification'])
training_error=[]
no_of_nods=[]
for i in range(5,20):
    set_zero()
    train_data,validation_data=divide(df,validation_data_size=0.2)
    tree=id3_tree(train_data,max_depth=i)
    acc=get_accuracy(train_data,tree)
    print("test:",i-4)
    no_of_nods.append(get_nodes())
    training_error.append(1-accuracy)


# #  Graph: Number of nodes vs Training Error 

# In[699]:


plt.plot(number_of_nods,training_error)
plt.xlabel("Number of Nodes")
plt.ylabel("Training Error")
plt.show()


# # 6.Handling missing attributes in test samples

# ### 1. Null Strategy
#  In Null Value strategy,‘Null’
# is considered as a special value both at training and testing
# time. 
# ### 2. Replacing with mean ,median or mode 
# 
#      This strategy can be applied on a feature which has numeric data like the height,width of a object or the price of artivle. We can calculate the mean, median or mode of the feature and replace it with the missing values (For training dataset). This is an approximation which can add variance to the data set. But the loss of the data can be negated by this method which yields better results compared to removal of rows and columns. Replacing with the above three approximations are a statistical approach of handling the missing values. This method is also called as leaking the data while training. 
#      Another way is to approximate it with the deviation of neighbouring values. This works better if the data is linear.when the attribute is missing from training dataset we can replace it with mean,median or mode and while testing we estimate the most common occuring value of that attribute i.e attribute value having highest probability.Missing values can be replaced by its distribution.
