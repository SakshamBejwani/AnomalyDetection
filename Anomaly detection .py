#!/usr/bin/env python
# coding: utf-8

# In[ ]:





# In[1]:


import cv2
import os
import random
import pickle
import numpy as np
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
from sklearn.model_selection import train_test_split
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import Xception
from tensorflow.keras.layers import Input,AveragePooling2D,Dropout,Flatten,Dense,BatchNormalization
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from sklearn.preprocessing import LabelBinarizer
from sklearn.metrics import classification_report
import datetime


# In[9]:


# Load the TensorBoard notebook extension
get_ipython().run_line_magic('load_ext', 'tensorboard')


# In[2]:


datadir = "C:\\Users\\snsha\\Anaconda3\\anaconda\\envs\\tensor-flow\\Anomaly detection\\Dataset"
cat = ["Non-violence", "Violence"]
training_data = []
for cate in cat:
    path = os.path.join(datadir,cate)
    class_num = cat.index(cate)
    for img in os.listdir(path):
        img_array = cv2.imread(os.path.join(path,img))
        filter = np.array([[-1,-1,-1],[-1,9,-1],[-1,-1,-1]])
        new_img = cv2.filter2D(img_array, -1, filter)
        new_array = cv2.resize(new_img,(299,299))
        new_array = cv2.cvtColor(new_array, cv2.COLOR_BGR2RGB)
        training_data.append([new_array,class_num])
random.shuffle(training_data) # shuffling the dataset 
X = []
Y = []
for features,labels in training_data:
    X.append(features)
    Y.append(labels)
#saving the dataset as pickle file 
pickle_out = open("C:\\Users\\snsha\\Anaconda3\\anaconda\\envs\\tensor-flow\\Anomaly detection\\X_1.pickle","wb")
pickle.dump(X, pickle_out)
pickle_out.close()
pickle_out = open("C:\\Users\\snsha\\Anaconda3\\anaconda\\envs\\tensor-flow\\Anomaly detection\\Y_1.pickle","wb")
pickle.dump(Y, pickle_out)
pickle_out.close()



# In[3]:


if os.path.getsize("C:\\Users\\snsha\\Anaconda3\\anaconda\\envs\\tensor-flow\\Anomaly detection\\X_1.pickle") > 0:
    with open("C:\\Users\\snsha\\Anaconda3\\anaconda\\envs\\tensor-flow\\Anomaly detection\X_1.pickle","rb") as f:
        unpickler = pickle.Unpickler(f)
        X = unpickler.load()
    with open("C:\\Users\\snsha\\Anaconda3\\anaconda\\envs\\tensor-flow\\Anomaly detection\\Y_1.pickle","rb") as f:
        unpickler = pickle.Unpickler(f)
        Y = unpickler.load()

#slpitting images into test and train sets
x_train_orig, x_test_orig, y_train_orig, y_test_orig = train_test_split(X, Y, test_size=0.15, random_state=1)
x_train = np.array(x_train_orig)
x_test = np.array(x_test_orig)
y_train = np.array(y_train_orig)
y_test = np.array(y_test_orig)


# In[ ]:





# In[4]:


#def convert_to_one_hot(labels, C):
    #C = tf.constant(C, name = "C")
    #one_hot_matrix = tf.one_hot(labels,C,axis = 0)
    #return one_hot_matrix
#y_train = convert_to_one_hot(y_train, 2)
#y_test = convert_to_one_hot(y_test, 2)
lb = LabelBinarizer()
y_train = lb.fit_transform(y_train)
y_test = lb.fit_transform(y_test)


# In[5]:


train_aug = ImageDataGenerator(
    rescale=1./255,
    rotation_range=30,
    zoom_range=0.15,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.15,
    horizontal_flip=True,
    fill_mode="nearest")

val_aug = ImageDataGenerator(rescale=1./255,fill_mode="nearest")


# In[ ]:





# In[6]:


#mean = np.array([123.68, 116.779, 103.939], dtype="float32")
#train_aug.mean = mean
#val_aug.mean = mean


# In[7]:


preModel = Xception(weights="imagenet", include_top=False, input_tensor=Input(shape=(299, 299, 3)))
preModel.trainable = False


# In[8]:



headModel = preModel.output
headModel = AveragePooling2D(pool_size=(7, 7))(headModel)
headModel = BatchNormalization(axis=1)(headModel)
headModel = Flatten(name="flatten")(headModel)
headModel = BatchNormalization(axis=1)(headModel)
headModel = Dense(512, activation="relu")(headModel)
headModel = BatchNormalization(axis=1)(headModel)
headModel = Dropout(0.5)(headModel)
headModel = Dense(256, activation="relu")(headModel)
headModel = BatchNormalization(axis=1)(headModel)
headModel = Dropout(0.3)(headModel)
headModel = Dense(128, activation="relu")(headModel)
headModel = BatchNormalization(axis=1)(headModel)
#headModel = Dropout(0.3)(headModel)
headModel = Dense(1, activation="sigmoid")(headModel)
  


# In[9]:


model = Model(inputs=preModel.input, outputs=headModel)
#model.summary()


# In[10]:


opt = Adam(lr=1e-3)
model.compile(loss="binary_crossentropy", optimizer=opt,metrics=["accuracy"])


# In[ ]:





# In[11]:


trained_model = model.fit(
    x = train_aug.flow(x_train, y_train, batch_size=32),
    steps_per_epoch=len(x_train) // 32,
    validation_data=val_aug.flow(x_test, y_test),
    validation_steps=len(x_test) // 32,
    epochs=30)


# In[ ]:


len(preModel.layers)


# In[ ]:



preModel.trainable = True
for layer in preModel.layers[112:]:
    layer.trainable = True


# In[ ]:


opt = Adam(lr=1e-4)
model.compile(loss="binary_crossentropy", optimizer=opt,metrics=["accuracy"])


# In[ ]:


fine_tune = 10
total = 30
hist = model.fit(
    x = train_aug.flow(x_train, y_train, batch_size=32),
    steps_per_epoch=len(x_train) // 32,
    validation_data=val_aug.flow(x_test, y_test),
    validation_steps=len(x_test) // 32,
    epochs=total,
    initial_epoch = 20)


# In[ ]:


predictions = model.predict(x=x_test.astype("float32"), batch_size=32)
predictions[51]


# In[ ]:


y_test[51]


# In[12]:


X_test = x_test/255.0
model.evaluate(X_test, y_test)


# In[ ]:


img_array = cv2.imread("/content/drive/My Drive/Colab Notebooks/img_7.jpg")
filter = np.array([[-1,-1,-1],[-1,9,-1],[-1,-1,-1]])
new_img = cv2.filter2D(img_array, -1, filter)
img_array = cv2.resize(img_array,(299,299))
preds = model.predict(np.expand_dims(img_array, axis=0))[0]
print(preds)
plt.imshow(new_img)


# In[24]:


N = 30
plt.style.use("ggplot")
plt.figure()
plt.plot(np.arange(0, N), trained_model.history["loss"], label="train_loss")
plt.plot(np.arange(0, N), trained_model.history["val_loss"], label="val_loss")
plt.plot(np.arange(0, N), trained_model.history["accuracy"], label="train_acc")
plt.plot(np.arange(0, N), trained_model.history["val_accuracy"], label="val_acc")
plt.title("Training Loss and Accuracy on Dataset")
plt.xlabel("Epoch #")
plt.ylabel("Loss/Accuracy")
plt.legend(loc="lower left")


# In[ ]:


N = 10
plt.style.use("ggplot")
plt.figure()
plt.plot(np.arange(0, N), hist.history["loss"], label="train_loss")
plt.plot(np.arange(0, N), hist.history["val_loss"], label="val_loss")
plt.plot(np.arange(0, N), hist.history["accuracy"], label="train_acc")
plt.plot(np.arange(0, N), hist.history["val_accuracy"], label="val_acc")
plt.title("Training Loss and Accuracy on Dataset")
plt.xlabel("Epoch #")
plt.ylabel("Loss/Accuracy")
plt.legend(loc="lower left")


# In[23]:


model.save("anomalydetection_Xception_new.h5")


# In[44]:


loaded_model = keras.models.load_model('anomalydetection_Xception_1.h5')
#loaded_model.summary()


# In[ ]:


img_array = cv2.imread("/content/drive/My Drive/Colab Notebooks/img_6.jpg")
img_array = cv2.resize(img_array,(299,299))
filter = np.array([[-1,-1,-1],[-1,10,-1],[-1,-1,-1]])
new_img = cv2.filter2D(img_array, -1, filter)

preds = loaded_model.predict(np.expand_dims(img_array, axis=0))[0]
if preds >= 0.5:
  print("Violence")
else:
  print("NON-violence")
  


# In[13]:


from collections import deque


# In[14]:


Q = deque(maxlen=64)


# In[15]:


import tensorflow
from tensorflow import keras
import cv2
import numpy as np


# In[30]:


vs = cv2.VideoCapture("C:\\Users\\snsha\\Anaconda3\\anaconda\\envs\\tensor-flow\\Anomaly detection\\videoplayback.mp4")
#vs = cv2.VideoCapture(0)
writer = None
(W, H) = (None, None)
label = None
output_path = 'result_xcep_realtime_4.mp4'
while True:
    (grabbed, frame) = vs.read()
    if not grabbed:
        break
    if W is None or H is None:
        (H, W) = frame.shape[:2]
        h = int(H)
        w = int(W)
    output = frame.copy()
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    frame = cv2.resize(frame, (299,299)).astype("float32")
    frame = frame/255.0
    preds = model.predict(np.expand_dims(frame, axis=0))[0]
    Q.append(preds)
    results = np.array(Q).mean(axis=0)
    if results[0] >= 0.5:
        label = "Violent"
        text = "{} activity".format(label)
        cv2.putText(output, text, (35, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.25, (0, 0, 255), 5)
    else:
        label = "Non-Violent"
        text = "{} activity".format(label)
        cv2.putText(output, text, (35, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.25, (0, 255, 0), 5)
    #text = "{} activity".format(label)
    #cv2.putText(output, text, (35, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.25, (0, 255, 0), 5)
    if writer is None:
        writer = cv2.VideoWriter(output_path, cv2.VideoWriter_fourcc(*'XIVD'), 10, (w, h), True)
    writer.write(output)
    cv2.imshow("window",output)
    key = cv2.waitKey(1) & 0xFF
    if key == ord("q"):
        break
writer.release()
cv2.destroyAllWindows() 


# In[8]:


from datetime import datetime


# In[9]:


def report(label,time):
    with open('C:\\Users\\snsha\\Anaconda3\\anaconda\\envs\\tensor-flow\\Anomaly detection\\report.csv','r+') as f:
               
        #now = datetime.now()
        dt = time.strftime('%H:%M:%S')
        f.writelines(f"\n{label},{time}")


# In[10]:


#to print into csv
vs = cv2.VideoCapture(0)
writer = None
(W, H) = (None, None)
label = None
while True:
    
    (grabbed, frame) = vs.read()
    if not grabbed:
        break
    frame_ori = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    frame = cv2.resize(frame_ori, (299,299)).astype("float32")
    frame_norm = frame/255.0
    preds = loaded_model.predict(np.expand_dims(frame_norm, axis=0))[0]
    Q.append(preds)
    results = np.array(Q).mean(axis=0)
    if results[0] >= 0.5:
        label = "Violent"
        i = datetime.now()
                
    else:
        label = "NOn-violent"
    date = datetime.date(1,1,1)
    f = datetime.now()
    d1 = datetime.combine(date, i)
    d2 = datetime.combine(date, f)
                 
    time = d2 - d1
    report(label,time)
    cv2.imshow("window",frame_ori)
    key = cv2.waitKey(1) & 0xFF
    if key == ord("q"):
        break
vs.release()


# In[ ]:


lst = [0,1,2,3]
for i in lst:
    if i == 2:
        brea

