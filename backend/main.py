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

import tensorflow as tf

print("✅ TensorFlow imported")

import numpy as np

print("✅ NumPy imported")

import tensorflow.keras.datasets.mnist as mnist

print("✅ Keras datasets imported")

import time
import os

print("✅ All imports completed")
print("🚀 Starting FastAPI application...")

# Create models directory
os.makedirs("models", exist_ok=True)

print("✅ Models directory created")

built_model = None

# # MULTIPROCESSING SOLUTION
# from multiprocessing import set_start_method
# from multiprocessing import Process, Manager
import os

# Reduce TensorFlow memory usage
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # Suppress TF logs
os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'
os.environ['TF_GPU_THREAD_MODE'] = 'gpu_private'
os.environ['OMP_NUM_THREADS'] = '1'

# # Set multiprocessing start method
# try:
#     set_start_method('spawn')
# except RuntimeError:
#     pass  # Already set

# Lazy Load
def load_mnist_data():
    print("📥 Loading MNIST data...")
    (x_train, y_train), (x_test, y_test) = mnist.load_data()
    print("✅ MNIST data loaded")

    print("🔧 Preprocessing data...")
    x_train = x_train / 255
    x_test = x_test / 255

    x_train = x_train.reshape((x_train.shape[0],28,28,1))
    x_test = x_test.reshape((x_test.shape[0],28,28,1))

    y_train = tf.keras.utils.to_categorical(y_train)
    y_test = tf.keras.utils.to_categorical(y_test)
    print("✅ Data preprocessing completed")
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
    greatest = num[0][0]
    greatest_num = 0
    for num_i in range(len(num[0])):
        comp = num[0][num_i]
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

    built_model = tf.keras.models.Sequential()

    prev_layer = None

    prev_nodes = 0

    built_model.add(tf.keras.Input(shape=(28,28,1)))

    for pos,layer in enumerate(model.layers):
        if layer.name == "Layer" and (prev_layer == "Convolutional Layer" or pos == 0):
            built_model.add(tf.keras.layers.Flatten())
        elif layer.name == "Convolutional Layer" and prev_layer == "Layer":
            built_model.add(tf.keras.layers.Reshape(target_shape=(prev_nodes,1,1)))

        if layer.name == "Convolutional Layer":
            built_model.add(tf.keras.layers.Conv2D(layer.layers, (4,4), activation="relu", padding="same"))
            if layer.pooling > 0:
                try: 
                    built_model.add(tf.keras.layers.MaxPool2D((layer.pooling,layer.pooling)))
                except:
                    raise PoolingError()
        elif layer.name == "Layer":
            built_model.add(tf.keras.layers.Dense(layer.layers, activation="relu"))
        else:
            print("Error: Unrecognized Layer")
        prev_layer = layer.name
        prev_nodes = layer.layers
    if prev_layer == "Convolutional Layer" or len(model.layers)==0:
        built_model.add(tf.keras.layers.Flatten())

    built_model.add(tf.keras.layers.Dense(classes, activation="softmax"))

    return built_model

def ParamLimit(value):
    if value >= 200000:
        return True
    return False

def fr_train_model(model, built_model, c_x_train, c_y_train):
    time_now = time.time()

    # if method == "pretrained":
    #     built_model.fit(x_train,y_train, epochs=model.epochs, batch_size=100, verbose=2)

    # elif method == "custom":
    #     built_model.fit(c_x_train,c_y_train, epochs=model.epochs, batch_size=100, verbose=2)

    built_model.fit(c_x_train,c_y_train, epochs=model.epochs, batch_size=100, verbose=2)
    
    return built_model, round(time.time() - time_now,3)

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
    (x_train, y_train), (x_test, y_test) = load_mnist_data()

    model_accuracy = None

    if (len(model.customData) == 0):

        try:
            built_model = build_model(model, "pretrained", None)
        except PoolingError as e:
            return [
                Stats(modelID="0", accuracy=0, parameters=0, trainingTime=0, error="Error during pooling")
            ]

        built_model.compile(metrics=["accuracy"], optimizer=tf.keras.optimizers.Adam(learning_rate=model.learningRate), loss=tf.keras.losses.CategoricalCrossentropy())

        too_many_params_premade = built_model.count_params()

        if ParamLimit(too_many_params_premade):
            error_params = "Model is too big, you have " + str(too_many_params_premade) + " parameters, try decreasing number of layers or other parameters. Please don't set our servers on fire :("
            return [
                Stats(modelID="0", accuracy=0, parameters=too_many_params_premade, trainingTime=0, error=error_params)
            ]

        built_model,tt = fr_train_model(model,built_model, x_train, y_train)

        model_accuracy = float(round(built_model.evaluate(x_test,y_test)[1],4) * 100)
        
    else:
        c_y_train = []
        c_x_train = []

        for instance in model.customData:
            c_x_train.append(instance.drawing)
            c_y_train.append(instance.label)

        c_x_train = np.array(c_x_train).reshape(len(c_x_train),28,28,1) / 255

        c_y_train,numclasses = fKeras(c_y_train)
        try:
            built_model = build_model(model, "custom", numclasses)
        except PoolingError as e:
            return [
                Stats(modelID="0", accuracy=0, parameters=0, trainingTime=0, error="Error during pooling")
            ]
        
        built_model.compile(metrics=["accuracy"], optimizer=tf.keras.optimizers.Adam(learning_rate=model.learningRate), loss=tf.keras.losses.CategoricalCrossentropy())

        too_many_params_custom = built_model.count_params()

        if ParamLimit(too_many_params_custom):
            error_params = "Model is too big, you have " + str(too_many_params_custom) + " parameters, try decreasing number of layers or other parameters. Please don't set our servers on fire :("
            return [
                Stats(modelID="0", accuracy=0, parameters=too_many_params_premade, trainingTime=0, error=error_params)
            ]

        c_y_train = np.array(c_y_train)

        built_model,tt = fr_train_model(model,built_model, c_x_train, c_y_train)

        #if model accuracy is 101 it means custom model was trained so accuracy not available
        model_accuracy = 101

    model_params = built_model.count_params()

    model_id = str(uuid.uuid4())
    model_name = "model-" + model_id + ".keras"

    built_model.save("models/" + model_name)

    #delete models older than 1 day
    for model_filename in os.listdir("models"):
        model_location = os.path.join("models", model_filename)
        model_time = os.path.getmtime(model_location)
        if(model_time < time.time() - 60*60*24):
            os.remove(model_location)

    return [
        Stats(modelID=model_id, accuracy=model_accuracy, parameters=model_params, trainingTime=tt, error="")
    ]

async def train_model_multiprocess(model: Model):
    manager = Manager()
    return_result = manager.dict()

    def train_process(res):
        (x_train, y_train), (x_test, y_test) = load_mnist_data()
        
        model_accuracy = None

        if (len(model.customData) == 0):

            try:
                built_model = build_model(model, "pretrained", None)
            except PoolingError as e:
                res['result'] = [
                    Stats(modelID="0", accuracy=0, parameters=0, trainingTime=0, error="Error during pooling")
                ]
                return

            built_model.compile(metrics=["accuracy"], optimizer=tf.keras.optimizers.Adam(learning_rate=model.learningRate), loss=tf.keras.losses.CategoricalCrossentropy())

            too_many_params_premade = built_model.count_params()

            if ParamLimit(too_many_params_premade):
                error_params = "Model is too big, you have " + str(too_many_params_premade) + " parameters, try decreasing number of layers or other parameters. Please don't set our servers on fire :("
                res['result'] = [
                    Stats(modelID="0", accuracy=0, parameters=too_many_params_premade, trainingTime=0, error=error_params)
                ]
                return

            built_model,tt = fr_train_model(model,built_model, x_train, y_train)

            model_accuracy = float(round(built_model.evaluate(x_test,y_test)[1],4) * 100)
            
        else:
            c_y_train = []
            c_x_train = []

            for instance in model.customData:
                c_x_train.append(instance.drawing)
                c_y_train.append(instance.label)

            c_x_train = np.array(c_x_train).reshape(len(c_x_train),28,28,1) / 255

            c_y_train,numclasses = fKeras(c_y_train)
            try:
                built_model = build_model(model, "custom", numclasses)
            except PoolingError as e:
                res['result'] = [
                    Stats(modelID="0", accuracy=0, parameters=0, trainingTime=0, error="Error during pooling")
                ]
                return
            
            built_model.compile(metrics=["accuracy"], optimizer=tf.keras.optimizers.Adam(learning_rate=model.learningRate), loss=tf.keras.losses.CategoricalCrossentropy())

            too_many_params_custom = built_model.count_params()

            if ParamLimit(too_many_params_custom):
                error_params = "Model is too big, you have " + str(too_many_params_custom) + " parameters, try decreasing number of layers or other parameters. Please don't set our servers on fire :("
                res['result'] = [
                    Stats(modelID="0", accuracy=0, parameters=too_many_params_premade, trainingTime=0, error=error_params)
                ]
                return

            c_y_train = np.array(c_y_train)

            built_model,tt = fr_train_model(model,built_model, c_x_train, c_y_train)

            #if model accuracy is 101 it means custom model was trained so accuracy not available
            model_accuracy = 101

        model_params = built_model.count_params()

        model_id = str(uuid.uuid4())
        model_name = "model-" + model_id + ".keras"

        built_model.save("models/" + model_name)

        #delete models older than 1 day
        for model_filename in os.listdir("models"):
            model_location = os.path.join("models", model_filename)
            model_time = os.path.getmtime(model_location)
            if(model_time < time.time() - 60*60*24):
                os.remove(model_location)

        res['result'] = [
            Stats(modelID=model_id, accuracy=model_accuracy, parameters=model_params, trainingTime=tt, error="")
        ]
        return
        # return [
        #     Stats(modelID=model_id, accuracy=model_accuracy, parameters=model_params, trainingTime=tt, error="")
        # ]
    
    p = Process(target=train_process, args=(return_result,))
    p.start()
    p.join()
    return return_result['result']

@app.post("/predict/")
async def predict(predict: Predict):
    pred_model_name = "model-" + predict.modelID + ".keras"
    loaded_model = tf.keras.models.load_model("models/" + pred_model_name)

    image = np.array(predict.predictImg).reshape((1,28,28,1))

    prediction =  str(predict.labels[FindNum(loaded_model.predict(image))])

    return [
        Prediction(prediction=prediction)
    ]

async def predict_multiprocess(predict: Predict):
    manager = Manager()
    return_result = manager.dict()

    def predict_process(res):

        pred_model_name = "model-" + predict.modelID + ".keras"
        loaded_model = tf.keras.models.load_model("models/" + pred_model_name)

        image = np.array(predict.predictImg).reshape((1,28,28,1))

        prediction =  str(predict.labels[FindNum(loaded_model.predict(image))])

        res['result'] = [
            Prediction(prediction=prediction)
        ]
        return
    
    p = Process(target=predict_process, args=(return_result,))
    p.start()
    p.join()
    return return_result['result']


# if __name__ == "__main__":
#     import uvicorn
#     port = int(os.environ.get("PORT", 8080))
#     print(f"🚀 Starting server on port {port}")
#     uvicorn.run(app, host="0.0.0.0", port=port)

print("🎉 All routes registered, app should be ready")














# from flask import Flask, request, jsonify
# from flask_cors import CORS
# import tensorflow as tf
# import numpy as np
# import tensorflow.keras.datasets.mnist as mnist
# import time
# import os
# import uuid

# print("🚀 Starting Flask application...")

# # Reduce TensorFlow logging
# os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

# app = Flask(__name__)
# CORS(app)

# # Create models directory
# os.makedirs("models", exist_ok=True)
# print("✅ Models directory created")

# # Your existing helper functions (keep all of them exactly as you have them)
# def unique(lst):
#     found = []
#     uni = 0
#     for pos,val in enumerate(lst):
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

# def build_model(model_data, type, numclasses):
#     classes = 10 if type == "pretrained" else numclasses
#     built_model = tf.keras.models.Sequential()
    
#     # ... your existing build_model logic ...
#     built_model.add(tf.keras.Input(shape=(28,28,1)))
    
#     prev_layer = None
#     prev_nodes = 0
    
#     for pos,layer in enumerate(model_data['layers']):
#         if layer['name'] == "Layer" and (prev_layer == "Convolutional Layer" or pos == 0):
#             built_model.add(tf.keras.layers.Flatten())
#         elif layer['name'] == "Convolutional Layer" and prev_layer == "Layer":
#             built_model.add(tf.keras.layers.Reshape(target_shape=(prev_nodes,1,1)))

#         if layer['name'] == "Convolutional Layer":
#             built_model.add(tf.keras.layers.Conv2D(layer['layers'], (4,4), activation="relu", padding="same"))
#             if layer['pooling'] > 0:
#                 built_model.add(tf.keras.layers.MaxPool2D((layer['pooling'],layer['pooling'])))
#         elif layer['name'] == "Layer":
#             built_model.add(tf.keras.layers.Dense(layer['layers'], activation="relu"))
        
#         prev_layer = layer['name']
#         prev_nodes = layer['layers']
    
#     if prev_layer == "Convolutional Layer" or len(model_data['layers'])==0:
#         built_model.add(tf.keras.layers.Flatten())

#     built_model.add(tf.keras.layers.Dense(classes, activation="softmax"))
#     return built_model

# def ParamLimit(value):
#     return value >= 200000

# def fr_train_model(model_data, built_model, c_x_train, c_y_train):
#     time_now = time.time()
#     built_model.fit(c_x_train, c_y_train, epochs=model_data['epochs'], batch_size=100, verbose=2)
#     return built_model, round(time.time() - time_now, 3)

# def load_mnist_data():
#     print("📥 Loading MNIST data...")
#     (x_train, y_train), (x_test, y_test) = mnist.load_data()
#     x_train = x_train / 255
#     x_test = x_test / 255
#     x_train = x_train.reshape((x_train.shape[0],28,28,1))
#     x_test = x_test.reshape((x_test.shape[0],28,28,1))
#     y_train = tf.keras.utils.to_categorical(y_train)
#     y_test = tf.keras.utils.to_categorical(y_test)
#     print("✅ MNIST data loaded and processed")
#     return (x_train, y_train), (x_test, y_test)

# @app.route('/')
# def root():
#     return {"message": "You were the chosen one..."}

# @app.route('/health')
# def health():
#     return {"status": "healthy", "timestamp": time.time()}

# @app.route('/train/', methods=['POST'])
# def train_model():
#     try:
#         model_data = request.get_json()
#         print("🎯 Received training request")
        
#         (x_train, y_train), (x_test, y_test) = load_mnist_data()
        
#         model_accuracy = None
        
#         if not model_data.get('customData'):
#             # Pretrained model logic
#             built_model = build_model(model_data, "pretrained", None)
#             built_model.compile(
#                 metrics=["accuracy"], 
#                 optimizer=tf.keras.optimizers.Adam(learning_rate=model_data['learningRate']), 
#                 loss=tf.keras.losses.CategoricalCrossentropy()
#             )
            
#             if ParamLimit(built_model.count_params()):
#                 return jsonify([{
#                     "modelID": "0", 
#                     "accuracy": 0, 
#                     "parameters": built_model.count_params(), 
#                     "trainingTime": 0, 
#                     "error": "Model too large"
#                 }])
            
#             built_model, tt = fr_train_model(model_data, built_model, x_train, y_train)
#             model_accuracy = float(round(built_model.evaluate(x_test, y_test)[1], 4) * 100)
#         else:
#             # Custom data logic
#             c_x_train = np.array([instance['drawing'] for instance in model_data['customData']]).reshape(len(model_data['customData']), 28, 28, 1) / 255
#             c_y_train = [instance['label'] for instance in model_data['customData']]
            
#             c_y_train, numclasses = fKeras(c_y_train)
#             built_model = build_model(model_data, "custom", numclasses)
#             built_model.compile(
#                 metrics=["accuracy"], 
#                 optimizer=tf.keras.optimizers.Adam(learning_rate=model_data['learningRate']), 
#                 loss=tf.keras.losses.CategoricalCrossentropy()
#             )
            
#             c_y_train = np.array(c_y_train)
#             built_model, tt = fr_train_model(model_data, built_model, c_x_train, c_y_train)
#             model_accuracy = 101
        
#         model_id = str(uuid.uuid4())
#         model_name = f"model-{model_id}.keras"
#         built_model.save(f"models/{model_name}")
        
#         # Clean up old models
#         for model_filename in os.listdir("models"):
#             model_path = os.path.join("models", model_filename)
#             if os.path.getmtime(model_path) < time.time() - 86400:  # 24 hours
#                 os.remove(model_path)
        
#         return jsonify([{
#             "modelID": model_id,
#             "accuracy": model_accuracy,
#             "parameters": built_model.count_params(),
#             "trainingTime": tt,
#             "error": ""
#         }])
        
#     except Exception as e:
#         print(f"❌ Training error: {e}")
#         return jsonify([{
#             "modelID": "0",
#             "accuracy": 0,
#             "parameters": 0,
#             "trainingTime": 0,
#             "error": str(e)
#         }])

# @app.route('/predict/', methods=['POST'])
# def predict():
#     try:
#         predict_data = request.get_json()
#         print("🎯 Received prediction request")
        
#         model_name = f"model-{predict_data['modelID']}.keras"
#         loaded_model = tf.keras.models.load_model(f"models/{model_name}")
        
#         image = np.array(predict_data['predictImg']).reshape((1, 28, 28, 1))
#         prediction_idx = FindNum(loaded_model.predict(image))
#         prediction = predict_data['labels'][prediction_idx]
        
#         return jsonify([{"prediction": prediction}])
        
#     except Exception as e:
#         print(f"❌ Prediction error: {e}")
#         return jsonify([{"prediction": f"Error: {str(e)}"}])

# # DEBUGGING ROUTE
# @app.route('/simple-test/', methods=['POST'])
# def simple_test():
#     try:
#         print("✅ Simple test endpoint reached")
#         return jsonify({"status": "simple test works", "timestamp": time.time()})
#     except Exception as e:
#         print(f"❌ Simple test error: {e}")
#         return jsonify({"error": str(e)}), 500

# # if __name__ == '__main__':
# #     port = int(os.environ.get('PORT', 8080))
# #     print(f"🚀 Starting Flask server on port {port}")
# #     app.run(host='0.0.0.0', port=port, debug=False)

# print("🎉 Flask app ready for production")

















# from flask import Flask, request, jsonify
# from flask_cors import CORS
# import os

# print("🚀 Starting minimal Flask app...")

# app = Flask(__name__)
# CORS(app)

# @app.route('/')
# def root():
#     return {"message": "You were the chosen one..."}

# @app.route('/health')
# def health():
#     return {"status": "healthy"}

# @app.route('/simple-test/', methods=['POST'])
# def simple_test():
#     print("✅ Simple test endpoint reached")
#     return jsonify({"status": "simple test works"})

# @app.route('/train/', methods=['POST'])
# def train_model():
#     try:
#         print("🎯 Training endpoint reached")
#         return jsonify([{
#             "modelID": "test123",
#             "accuracy": 95.5,
#             "parameters": 1000,
#             "trainingTime": 5.2,
#             "error": ""
#         }])
#     except Exception as e:
#         return jsonify([{
#             "modelID": "0",
#             "accuracy": 0,
#             "parameters": 0,
#             "trainingTime": 0,
#             "error": str(e)
#         }]), 500

# if __name__ == '__main__':
#     port = int(os.environ.get('PORT', 8080))
#     print(f"🚀 Starting Flask server on port {port}")
#     app.run(host='0.0.0.0', port=port, debug=False)

