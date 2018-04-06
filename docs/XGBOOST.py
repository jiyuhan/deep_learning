
# coding: utf-8

# In[1]:


from numpy import loadtxt
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler
from time import time

scaler = StandardScaler()


# In[2]:


data = loadtxt('diabetes.csv', delimiter=",")


# In[3]:


X = data[:,0:8]
Y = data[:,8]
X = scaler.fit_transform(X)


# In[4]:


seed = 7
test_size = 0.33
x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=test_size, random_state=seed)


# In[5]:


from sklearn.svm import SVC


# In[6]:


svm = SVC(probability=False, kernel='rbf', C=2,  gamma=.0073)


# In[7]:


svm.fit(x_train, y_train)


# In[8]:


predicted = svm.predict(x_test)
print("Accuracy: %0.4f" % accuracy_score(y_test,
                                                     predicted))


# In[9]:


from xgboost import XGBClassifier


# In[10]:


xg = XGBClassifier()
xg.fit(x_train, y_train)


# In[11]:


predictions_xgb = xg.predict(x_test)


# In[12]:


print("Accuracy: %0.4f" % accuracy_score(y_test,
                                                     predictions_xgb))


# In[13]:


import keras
from keras.models import Sequential
from keras.layers import Dense
from keras.callbacks import TensorBoard


# In[14]:


model = Sequential()
model.add(Dense(500, input_dim=8, activation='relu'))
model.add(Dense(1, activation='sigmoid'))


# In[15]:


model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])


# In[16]:


model.fit(x_train, y_train, nb_epoch=50, batch_size=10, verbose=2)


# In[17]:


scores = model.evaluate(x_test, y_test)
print("%s: %.2f%%" % (model.metrics_names[1], scores[1]*100))


# In[18]:


import h5py
model.save('diabetes.h5')


# In[19]:


import pandas as pd
import tensorflow as tf
CSV_COLUMNS = ('id', 'dr', 'age', 'blood', 'gender', 'organ',
               'ethnicity', 'bmi', 'acceptance')


# In[20]:


input_reader = pd.read_csv(tf.gfile.Open('data-plz.csv'), names=CSV_COLUMNS, na_values=' ?')


# In[21]:


input_reader = input_reader.drop(['id'], axis=1)


# In[22]:


input_reader = input_reader.fillna(0)


# In[23]:


input_reader


# In[24]:


input_reader = input_reader.drop(input_reader.index[0])


# In[25]:


input_reader.blood = pd.Categorical(input_reader.blood)
input_reader['blood'] = input_reader.blood.cat.codes


# In[26]:


input_reader.dr = pd.Categorical(input_reader.dr)
input_reader['dr'] = input_reader.dr.cat.codes
input_reader.gender = pd.Categorical(input_reader.gender)
input_reader['gender'] = input_reader.gender.cat.codes
input_reader.ethnicity = pd.Categorical(input_reader.ethnicity)
input_reader['ethnicity'] = input_reader.ethnicity.cat.codes
# input_reader.acceptance = pd.Categorical(input_reader.acceptance)
# input_reader['acceptance'] = input_reader.acceptance.cat.codes
input_reader.organ = pd.Categorical(input_reader.organ)
input_reader['organ'] = input_reader.organ.cat.codes


# In[27]:


# input_reader = input_reader.drop(['organ'], axis=1)
# input_reader = input_reader.drop(['bmi'], axis=1)


# In[39]:


#input_reader = input_reader.apply(lambda col: pd.factorize(col)[0])


# In[29]:


y = input_reader['acceptance'].tolist()


# In[34]:


x = input_reader[['dr', 'age', 'blood', 'gender','organ','ethnicity', 'bmi']].as_matrix()
#x = scaler.fit_transform(x)


# In[31]:


seed = 7
test_size = 0.33
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=test_size, random_state=seed)
x_train


# In[32]:


learning_rate = 0.01
model = Sequential()
model.add(Dense(2000, input_dim=7, activation='relu'))
model.add(Dense(1, activation='sigmoid'))
model.compile(loss='binary_crossentropy',
                optimizer='sgd',
                metrics=['accuracy'])


# In[33]:


tensorboard = TensorBoard(log_dir="logs/{}".format(time()))
model.fit(x_train, y_train, nb_epoch=50, batch_size=100, verbose=1, callbacks=[tensorboard])


# In[ ]:


scores = model.evaluate(x_test, y_test)
print("%s: %.2f%%" % (model.metrics_names[1], scores[1]*100))


# In[ ]:


x_test[0]


# In[ ]:


svm = SVC(probability=False, kernel='rbf', C=2,  gamma=.0073)
svm.fit(x_train, y_train)
predicted = svm.predict(x_test)
print("Accuracy: %0.4f" % accuracy_score(y_test,
                                                     predicted))


# In[ ]:


xg = XGBClassifier()
xg.fit(x_train, y_train)
predictions_xgb = xg.predict(x_test)
print("Accuracy: %0.4f" % accuracy_score(y_test,
                                                     predictions_xgb))


# In[ ]:


model.save('transplant.h5')


# In[35]:


from keras.models import load_model
model = load_model('transplant.h5')


# In[38]:


x_test[0].reshape(1,7)


# In[36]:


model.predict(x_test[0].reshape((1,7)))


# In[ ]:


keras.utils.print_summary(model, line_length=None, positions=None, print_fn=None)


# In[ ]:


from tensorflow.python.saved_model import builder as saved_model_builder
from tensorflow.python.saved_model import tag_constants, signature_constants
from keras import backend as K
from tensorflow.python.saved_model.signature_def_utils_impl import predict_signature_def


# In[ ]:


builder = saved_model_builder.SavedModelBuilder('./output')
signature = predict_signature_def(inputs={'input': model.inputs[0]},
                                    outputs={'income': model.outputs[0]})
with K.get_session() as sess:
    builder.add_meta_graph_and_variables(
        sess=sess,
        tags=[tag_constants.SERVING],
        signature_def_map={
            signature_constants.DEFAULT_SERVING_SIGNATURE_DEF_KEY: signature}
    )
    builder.save()

