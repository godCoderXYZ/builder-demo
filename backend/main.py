print("🚀 Script starting...")

import ssl
ssl._create_default_https_context = ssl._create_unverified_context

print("✅ SSL context set")

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

print("✅ FastAPI imports done")

from pydantic import BaseModel
from typing import List
import uuid

print("✅ More imports done")

# Replace TensorFlow with scikit-learn
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import LabelBinarizer
from sklearn.metrics import accuracy_score
import joblib

print("✅ scikit-learn imported")

import numpy as np

print("✅ NumPy imported")

# # Replace Keras datasets with direct MNIST loading
# from sklearn.datasets import fetch_openml
# print("✅ sklearn datasets imported")

import time
import os

print("✅ All imports completed")
print("🚀 Starting FastAPI application...")

# Create models directory
os.makedirs("models", exist_ok=True)

print("✅ Models directory created")

built_model = None

import os

# scikit-learn doesn't need these environment variables, but keep them for compatibility
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # Suppress TF logs
os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'
os.environ['TF_GPU_THREAD_MODE'] = 'gpu_private'
os.environ['OMP_NUM_THREADS'] = '1'

# Lazy Load
def load_mnist_data():
    print("📥 Loading MNIST data from local .npy files...")
    
    try:
        # Load from your pre-downloaded .npy files
        x_train = np.load('mnist_data/x_train.npy')
        y_train = np.load('mnist_data/y_train.npy')
        x_test = np.load('mnist_data/x_test.npy')
        y_test = np.load('mnist_data/y_test.npy')
        
        print("✅ MNIST data loaded from local files")
        print(f"📊 Training data shape: {x_train.shape}")
        print(f"📊 Test data shape: {x_test.shape}")
        print(f"📊 Training labels shape: {y_train.shape}")
        print(f"📊 Test labels shape: {y_test.shape}")
        
        return (x_train, y_train), (x_test, y_test)
        
    except FileNotFoundError as e:
        print(f"❌ Local MNIST files not found: {e}")
        print("📥 Generating synthetic MNIST-like data as fallback...")
        
        # Fallback to synthetic data if files don't exist
        num_train = 1000
        num_test = 200
        
        x_train = np.random.rand(num_train, 28, 28, 1)
        x_test = np.random.rand(num_test, 28, 28, 1)
        
        y_train = np.random.randint(0, 10, num_train)
        y_test = np.random.randint(0, 10, num_test)
        
        # Convert to one-hot encoding for scikit-learn
        lb = LabelBinarizer()
        y_train = lb.fit_transform(y_train)
        y_test = lb.transform(y_test)
        
        print("✅ Synthetic MNIST data generated as fallback")
        return (x_train, y_train), (x_test, y_test)

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    # allow_origins=[
    #     "https://deep-dive-into-ai.vercel.app", # for production
    #     "http://localhost:5173",                # Vite dev server
    #     "http://localhost:4173",                # Vite preview
    # ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

print("✅ FastAPI app created successfully")

(x_train, y_train), (x_test, y_test) = load_mnist_data()

#SUBCLASS
#each layer will be inputted as this, as part of an array of layers in Model
class Layer(BaseModel):
    #the name is a string like "Layer" or "Convolutional Layer", case sensitive
    name: str
    #the number of layers or nodes for that layer as a number, example: 32
    layers: int
    #pooling: int by int
    pooling: int
#SUBCLASS
#the custom data will be submitted as an array of these, each object has a label and drawing
class CustomData(BaseModel):
    #the drawing is submitted encoded as a pixels list
    drawing: List[int]
    #the label is a string that identifies the submitted image, example: cat
    label: str

#INPUTS:

#this is the input for /train, which includes all the parameters for training
class Model(BaseModel):
    #array of layers derived from the blocks
    layers: List[Layer]
    #number of epochs for training, as an int, example: 2
    epochs: int
    #learning rate for training, as a float, example: 0.001
    learningRate: float
    #the custom data for the model training
    #if the array is [] or null, use pre-made number data
    customData: List[CustomData]

#this is the input for /predict
class Predict(BaseModel):
    #the unique id of the model used for predicting
    modelID: str
    #the image that needs to be predicted, encoded as a pixels list
    predictImg: List[int]
    #the list of labels, in order of first appearance
    labels: List[str]

#RETURNS:

#api will return these stats after training the model
class Stats(BaseModel):
    #the unique id of the model, can be used to access the model later
    modelID: str
    #accuracy as a number, not including the %, example: 96.5
    accuracy: float
    #number of parameters as a number, example: 7000
    parameters: int
    #training time in seconds, example: 9
    trainingTime: float
    #the error
    error: str

#api will return the prediction from the /predict
class Prediction(BaseModel):
    #prediction, as an str, example: cat
    prediction: str

# app = FastAPI()

class PoolingError(Exception):
    def __init__(self, message="This is a custom exception"):
        super().__init__(message)

def unique(list):
    found = []
    uni = 0
    for pos,val in enumerate(list):
        if val not in found:
            uni += 1
            found.append(val)
    return uni

def fKeras(labels):
    ret_mat = []
    classes = unique(labels)
    found = []
    for i,val in enumerate(labels):
        if val not in found:
            found.append(val)
    for i,val in enumerate(labels):
        temp = []
        for pos,string in enumerate(found):
            if string == val:
                temp.append(1)
            else:
                temp.append(0)
        ret_mat.append(temp)
    return ret_mat, classes

def FindNum(num):
    greatest = num[0]
    greatest_num = 0
    for num_i in range(len(num)):
        comp = num[num_i]
        if comp > greatest:
            greatest_num = num_i
            greatest = comp
    return greatest_num

def build_model(model: Model, type, numclasses):
    classes = None

    if type == "pretrained":
        classes = 10
    elif type == "custom":
        classes = numclasses

    # Convert layer configuration to scikit-learn format
    hidden_layer_sizes = []
    
    for layer in model.layers:
        if layer.name == "Layer":
            hidden_layer_sizes.append(layer.layers)
        elif layer.name == "Convolutional Layer":
            # sklearn doesn't support conv layers directly, approximate with dense
            hidden_layer_sizes.append(layer.layers)
    
    # Default hidden layers if none specified
    if not hidden_layer_sizes:
        hidden_layer_sizes = [64, 32]
    
    # Create MLP classifier
    built_model = MLPClassifier(
        hidden_layer_sizes=hidden_layer_sizes,
        max_iter=model.epochs,
        learning_rate_init=model.learningRate,
        batch_size=100,
        verbose=True
    )
    
    return built_model

def ParamLimit(value):
    if value >= 200000:
        return True
    return False

def fr_train_model(model, built_model, c_x_train, c_y_train):
    time_now = time.time()
    
    # Flatten images for sklearn (from 28x28x1 to 784)
    if len(c_x_train.shape) > 2:
        c_x_train_flat = c_x_train.reshape(c_x_train.shape[0], -1)
    else:
        c_x_train_flat = c_x_train
    
    built_model.fit(c_x_train_flat, c_y_train)
    
    return built_model, round(time.time() - time_now, 3)

@app.get("/")
async def root():
    return {"message": "You were the chosen one..."}

# Miscellaneous Routes
@app.get("/health")
async def health_check():
    return {"status": "healthy", "timestamp": time.time()}

@app.on_event("startup")
async def startup_event():
    print("✅ FastAPI startup complete - app is ready to receive requests")

@app.on_event("shutdown") 
async def shutdown_event():
    print("🛑 FastAPI shutting down")

@app.post("/train/")
async def train_model(model: Model):
    # (x_train, y_train), (x_test, y_test) = load_mnist_data()

    model_accuracy = None

    if (len(model.customData) == 0):
        try:
            built_model = build_model(model, "pretrained", None)
        except PoolingError as e:
            return [
                Stats(modelID="0", accuracy=0, parameters=0, trainingTime=0, error="Error during pooling")
            ]

        built_model, tt = fr_train_model(model, built_model, x_train, y_train)

        # sklearn models don't need compile step
        too_many_params_premade = sum(coef.size for coef in built_model.coefs_) + sum(intercept.size for intercept in built_model.intercepts_)

        if ParamLimit(too_many_params_premade):
            error_params = "Model is too big, you have " + str(too_many_params_premade) + " parameters, try decreasing number of layers or other parameters. Please don't set our servers on fire :("
            return [
                Stats(modelID="0", accuracy=0, parameters=too_many_params_premade, trainingTime=0, error=error_params)
            ]

        # Calculate accuracy - FIXED
        x_test_flat = x_test.reshape(x_test.shape[0], -1)

        # Use predict_proba and then argmax to get multiclass predictions
        y_pred_proba = built_model.predict_proba(x_test_flat)
        y_pred_multiclass = np.argmax(y_pred_proba, axis=1)

        # Convert y_test from one-hot to multiclass
        y_test_multiclass = np.argmax(y_test, axis=1)

        model_accuracy = float(accuracy_score(y_test_multiclass, y_pred_multiclass) * 100)
        
    else:
        c_y_train = []
        c_x_train = []

        for instance in model.customData:
            c_x_train.append(instance.drawing)
            c_y_train.append(instance.label)

        c_x_train = np.array(c_x_train).reshape(len(c_x_train), 28, 28, 1) / 255

        c_y_train, numclasses = fKeras(c_y_train)
        try:
            built_model = build_model(model, "custom", numclasses)
        except PoolingError as e:
            return [
                Stats(modelID="0", accuracy=0, parameters=0, trainingTime=0, error="Error during pooling")
            ]
        
        # sklearn models don't need compile step
        too_many_params_custom = sum(layer.size for layer in built_model.coefs_) + sum(layer.size for layer in built_model.intercepts_)

        if ParamLimit(too_many_params_custom):
            error_params = "Model is too big, you have " + str(too_many_params_custom) + " parameters, try decreasing number of layers or other parameters. Please don't set our servers on fire :("
            return [
                Stats(modelID="0", accuracy=0, parameters=too_many_params_custom, trainingTime=0, error=error_params)
            ]

        c_y_train = np.array(c_y_train)

        built_model, tt = fr_train_model(model, built_model, c_x_train, c_y_train)

        #if model accuracy is 101 it means custom model was trained so accuracy not available
        model_accuracy = 101

    model_params = sum(layer.size for layer in built_model.coefs_) + sum(layer.size for layer in built_model.intercepts_)

    model_id = str(uuid.uuid4())
    model_name = "model-" + model_id + ".joblib"  # Change extension for joblib

    # Save model using joblib
    joblib.dump(built_model, "models/" + model_name)

    #delete models older than 1 day
    for model_filename in os.listdir("models"):
        model_location = os.path.join("models", model_filename)
        model_time = os.path.getmtime(model_location)
        if(model_time < time.time() - 60*60*24):
            os.remove(model_location)

    return [
        Stats(modelID=model_id, accuracy=model_accuracy, parameters=model_params, trainingTime=tt, error="")
    ]

@app.post("/predict/")
async def predict(predict: Predict):
    pred_model_name = "model-" + predict.modelID + ".joblib"  # Change extension
    loaded_model = joblib.load("models/" + pred_model_name)

    image = np.array(predict.predictImg).reshape((1, 28, 28, 1))
    image_flat = image.reshape(1, -1)  # Flatten for sklearn

    prediction_probs = loaded_model.predict_proba(image_flat)[0]
    prediction_idx = FindNum(prediction_probs)
    prediction = str(predict.labels[prediction_idx])

    return [
        Prediction(prediction=prediction)
    ]

print("🎉 All routes registered, app should be ready")

















# print("🚀 Script starting...")

# import ssl
# ssl._create_default_https_context = ssl._create_unverified_context

# print("✅ SSL context set")

# from fastapi import FastAPI
# from fastapi.middleware.cors import CORSMiddleware

# print("✅ FastAPI imports done")

# from pydantic import BaseModel
# from typing import List
# import uuid

# print("✅ More imports done")

# import tensorflow as tf

# print("✅ TensorFlow imported")

# import numpy as np

# print("✅ NumPy imported")

# import tensorflow.keras.datasets.mnist as mnist

# print("✅ Keras datasets imported")

# import time
# import os

# print("✅ All imports completed")
# print("🚀 Starting FastAPI application...")

# # Create models directory
# os.makedirs("models", exist_ok=True)

# print("✅ Models directory created")

# built_model = None

# import os

# # Reduce TensorFlow memory usage
# os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # Suppress TF logs
# os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'
# os.environ['TF_GPU_THREAD_MODE'] = 'gpu_private'
# os.environ['OMP_NUM_THREADS'] = '1'

# # Lazy Load
# def load_mnist_data():
#     print("📥 Loading MNIST data...")
#     (x_train, y_train), (x_test, y_test) = mnist.load_data()
#     print("✅ MNIST data loaded")

#     print("🔧 Preprocessing data...")
#     x_train = x_train / 255
#     x_test = x_test / 255

#     x_train = x_train.reshape((x_train.shape[0],28,28,1))
#     x_test = x_test.reshape((x_test.shape[0],28,28,1))

#     y_train = tf.keras.utils.to_categorical(y_train)
#     y_test = tf.keras.utils.to_categorical(y_test)
#     print("✅ Data preprocessing completed")
#     return (x_train, y_train), (x_test, y_test)

# app = FastAPI()

# app.add_middleware(
#     CORSMiddleware,
#     allow_origins=["*"],
#     # allow_origins=[
#     #     "https://deep-dive-into-ai.vercel.app", # for production
#     #     "http://localhost:5173",                # Vite dev server
#     #     "http://localhost:4173",                # Vite preview
#     # ],
#     allow_credentials=True,
#     allow_methods=["*"],
#     allow_headers=["*"],
# )

# print("✅ FastAPI app created successfully")

# (x_train, y_train), (x_test, y_test) = load_mnist_data()

# #SUBCLASS
# #each layer will be inputted as this, as part of an array of layers in Model
# class Layer(BaseModel):
#     #the name is a string like "Layer" or "Convolutional Layer", case sensitive
#     name: str
#     #the number of layers or nodes for that layer as a number, example: 32
#     layers: int
#     #pooling: int by int
#     pooling: int
# #SUBCLASS
# #the custom data will be submitted as an array of these, each object has a label and drawing
# class CustomData(BaseModel):
#     #the drawing is submitted encoded as a pixels list
#     drawing: List[int]
#     #the label is a string that identifies the submitted image, example: cat
#     label: str

# #INPUTS:

# #this is the input for /train, which includes all the parameters for training
# class Model(BaseModel):
#     #array of layers derived from the blocks
#     layers: List[Layer]
#     #number of epochs for training, as an int, example: 2
#     epochs: int
#     #learning rate for training, as a float, example: 0.001
#     learningRate: float
#     #the custom data for the model training
#     #if the array is [] or null, use pre-made number data
#     customData: List[CustomData]

# #this is the input for /predict
# class Predict(BaseModel):
#     #the unique id of the model used for predicting
#     modelID: str
#     #the image that needs to be predicted, encoded as a pixels list
#     predictImg: List[int]
#     #the list of labels, in order of first appearance
#     labels: List[str]

# #RETURNS:

# #api will return these stats after training the model
# class Stats(BaseModel):
#     #the unique id of the model, can be used to access the model later
#     modelID: str
#     #accuracy as a number, not including the %, example: 96.5
#     accuracy: float
#     #number of parameters as a number, example: 7000
#     parameters: int
#     #training time in seconds, example: 9
#     trainingTime: float
#     #the error
#     error: str

# #api will return the prediction from the /predict
# class Prediction(BaseModel):
#     #prediction, as an str, example: cat
#     prediction: str

# # app = FastAPI()

# class PoolingError(Exception):
#     def __init__(self, message="This is a custom exception"):
#         super().__init__(message)

# def unique(list):
#     found = []
#     uni = 0
#     for pos,val in enumerate(list):
#         if val not in found:
#             uni += 1
#             found.append(val)
#     return uni

# def fKeras(labels):
#     ret_mat = []
#     classes = unique(labels)
#     found = []
#     for i,val in enumerate(labels):
#         if val not in found:
#             found.append(val)
#     for i,val in enumerate(labels):
#         temp = []
#         for pos,string in enumerate(found):
#             if string == val:
#                 temp.append(1)
#             else:
#                 temp.append(0)
#         ret_mat.append(temp)
#     return ret_mat, classes

# def FindNum(num):
#     greatest = num[0][0]
#     greatest_num = 0
#     for num_i in range(len(num[0])):
#         comp = num[0][num_i]
#         if comp > greatest:
#             greatest_num = num_i
#             greatest = comp
#     return greatest_num

# def build_model(model: Model, type, numclasses):

#     classes = None

#     if type == "pretrained":
#         classes = 10

#     elif type == "custom":
#         classes = numclasses

#     built_model = tf.keras.models.Sequential()

#     prev_layer = None

#     prev_nodes = 0

#     built_model.add(tf.keras.Input(shape=(28,28,1)))

#     for pos,layer in enumerate(model.layers):
#         if layer.name == "Layer" and (prev_layer == "Convolutional Layer" or pos == 0):
#             built_model.add(tf.keras.layers.Flatten())
#         elif layer.name == "Convolutional Layer" and prev_layer == "Layer":
#             built_model.add(tf.keras.layers.Reshape(target_shape=(prev_nodes,1,1)))

#         if layer.name == "Convolutional Layer":
#             built_model.add(tf.keras.layers.Conv2D(layer.layers, (4,4), activation="relu", padding="same"))
#             if layer.pooling > 0:
#                 try: 
#                     built_model.add(tf.keras.layers.MaxPool2D((layer.pooling,layer.pooling)))
#                 except:
#                     raise PoolingError()
#         elif layer.name == "Layer":
#             built_model.add(tf.keras.layers.Dense(layer.layers, activation="relu"))
#         else:
#             print("Error: Unrecognized Layer")
#         prev_layer = layer.name
#         prev_nodes = layer.layers
#     if prev_layer == "Convolutional Layer" or len(model.layers)==0:
#         built_model.add(tf.keras.layers.Flatten())

#     built_model.add(tf.keras.layers.Dense(classes, activation="softmax"))

#     return built_model

# def ParamLimit(value):
#     if value >= 200000:
#         return True
#     return False

# def fr_train_model(model, built_model, c_x_train, c_y_train):
#     time_now = time.time()

#     built_model.fit(c_x_train,c_y_train, epochs=model.epochs, batch_size=100, verbose=2)
    
#     return built_model, round(time.time() - time_now,3)

# @app.get("/")
# async def root():
#     return {"message": "You were the chosen one..."}

# # Miscellaneous Routes
# @app.get("/health")
# async def health_check():
#     return {"status": "healthy", "timestamp": time.time()}

# @app.on_event("startup")
# async def startup_event():
#     print("✅ FastAPI startup complete - app is ready to receive requests")

# @app.on_event("shutdown") 
# async def shutdown_event():
#     print("🛑 FastAPI shutting down")

# @app.post("/train/")
# async def train_model(model: Model):
#     # (x_train, y_train), (x_test, y_test) = load_mnist_data()

#     model_accuracy = None

#     if (len(model.customData) == 0):

#         try:
#             built_model = build_model(model, "pretrained", None)
#         except PoolingError as e:
#             return [
#                 Stats(modelID="0", accuracy=0, parameters=0, trainingTime=0, error="Error during pooling")
#             ]

#         built_model.compile(metrics=["accuracy"], optimizer=tf.keras.optimizers.Adam(learning_rate=model.learningRate), loss=tf.keras.losses.CategoricalCrossentropy())

#         too_many_params_premade = built_model.count_params()

#         if ParamLimit(too_many_params_premade):
#             error_params = "Model is too big, you have " + str(too_many_params_premade) + " parameters, try decreasing number of layers or other parameters. Please don't set our servers on fire :("
#             return [
#                 Stats(modelID="0", accuracy=0, parameters=too_many_params_premade, trainingTime=0, error=error_params)
#             ]

#         built_model,tt = fr_train_model(model,built_model, x_train, y_train)

#         model_accuracy = float(round(built_model.evaluate(x_test,y_test)[1],4) * 100)
        
#     else:
#         c_y_train = []
#         c_x_train = []

#         for instance in model.customData:
#             c_x_train.append(instance.drawing)
#             c_y_train.append(instance.label)

#         c_x_train = np.array(c_x_train).reshape(len(c_x_train),28,28,1) / 255

#         c_y_train,numclasses = fKeras(c_y_train)
#         try:
#             built_model = build_model(model, "custom", numclasses)
#         except PoolingError as e:
#             return [
#                 Stats(modelID="0", accuracy=0, parameters=0, trainingTime=0, error="Error during pooling")
#             ]
        
#         built_model.compile(metrics=["accuracy"], optimizer=tf.keras.optimizers.Adam(learning_rate=model.learningRate), loss=tf.keras.losses.CategoricalCrossentropy())

#         too_many_params_custom = built_model.count_params()

#         if ParamLimit(too_many_params_custom):
#             error_params = "Model is too big, you have " + str(too_many_params_custom) + " parameters, try decreasing number of layers or other parameters. Please don't set our servers on fire :("
#             return [
#                 Stats(modelID="0", accuracy=0, parameters=too_many_params_premade, trainingTime=0, error=error_params)
#             ]

#         c_y_train = np.array(c_y_train)

#         built_model,tt = fr_train_model(model,built_model, c_x_train, c_y_train)

#         #if model accuracy is 101 it means custom model was trained so accuracy not available
#         model_accuracy = 101

#     model_params = built_model.count_params()

#     model_id = str(uuid.uuid4())
#     model_name = "model-" + model_id + ".keras"

#     built_model.save("models/" + model_name)

#     #delete models older than 1 day
#     for model_filename in os.listdir("models"):
#         model_location = os.path.join("models", model_filename)
#         model_time = os.path.getmtime(model_location)
#         if(model_time < time.time() - 60*60*24):
#             os.remove(model_location)

#     return [
#         Stats(modelID=model_id, accuracy=model_accuracy, parameters=model_params, trainingTime=tt, error="")
#     ]

# @app.post("/predict/")
# async def predict(predict: Predict):
#     pred_model_name = "model-" + predict.modelID + ".keras"
#     loaded_model = tf.keras.models.load_model("models/" + pred_model_name)

#     image = np.array(predict.predictImg).reshape((1,28,28,1))

#     prediction =  str(predict.labels[FindNum(loaded_model.predict(image))])

#     return [
#         Prediction(prediction=prediction)
#     ]

# print("🎉 All routes registered, app should be ready")