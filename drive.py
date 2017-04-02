import argparse
import base64
from datetime import datetime
import os
import shutil

import numpy as np
import socketio
import eventlet
import eventlet.wsgi
from PIL import Image
from flask import Flask
from io import BytesIO

from keras.models import load_model
import h5py
from keras import __version__ as keras_version

import kanet as kn
import modelconfig as config

sio = socketio.Server()
app = Flask(__name__)
model = None
prev_image_array = None


class SimplePIController:
    def __init__(self, Kp, Ki, mode = 0):
        self.Kp = Kp
        self.Ki = Ki
        self.set_point = 0.
        self.error = 0.
        self.integral = 0.
        self.mode = mode
    
    def set_desired(self, desired):
        self.set_point = desired

    def update(self, throttle, measurement):
        self.error = self.set_point - measurement
        new_throttle = throttle
        # if accelerating passed desired speed then apply, otherwise use model
        if (measurement > self.set_point or self.mode == 0):
            # integral error
            self.integral += self.error
            
            new_throttle = self.Kp * self.error + self.Ki * self.integral
        
        return new_throttle


controller = SimplePIController(0.1, 0.002, mode = 1) # use mode=1 for model acceleration
set_speed = 20
controller.set_desired(set_speed)
max_throttle = 0.8 # 0.3

@sio.on('telemetry')
def telemetry(sid, data):
    if data:
        # The current steering angle of the car
        steering_angle = data["steering_angle"]
        # The current throttle of the car
        throttle = data["throttle"]
        # The current speed of the carKa
        speed = data["speed"]
        # The current image from the center camera of the car
        imgString = data["image"]
        image = Image.open(BytesIO(base64.b64decode(imgString)))
        image_array = np.asarray(image)
        #print("Predicting...")
        prediction = model.predict(image_array[None, :, :, :], batch_size=1)[0]
        #print("Prediction = {}".format(prediction))
        steering_angle = float(prediction[0])
        throttle = np.clip(float(prediction[1]), -max_throttle, max_throttle)

        throttle = controller.update(throttle, float(speed))

        print("Steering angle = {:.3f}   Throttle = {:.3f}   Speed = {:.2f}".format(steering_angle, throttle, float(speed)))
        send_control(steering_angle, throttle)

        # save frame
        if args.image_folder != '':
            timestamp = datetime.utcnow().strftime('%Y_%m_%d_%H_%M_%S_%f')[:-3]
            image_filename = os.path.join(args.image_folder, timestamp)
            image.save('{}.jpg'.format(image_filename))
    else:
        # NOTE: DON'T EDIT THIS.
        sio.emit('manual', data={}, skip_sid=True)


@sio.on('connect')
def connect(sid, environ):
    print("connect ", sid)
    send_control(0, 0)


def send_control(steering_angle, throttle):
    sio.emit(
        "steer",
        data={
            'steering_angle': steering_angle.__str__(),
            'throttle': throttle.__str__()
        },
        skip_sid=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Remote Driving')
    parser.add_argument(
        'model',
        type=str,
        help='Path to the model weights h5 file.'
    )
    parser.add_argument(
        'image_folder',
        type=str,
        nargs='?',
        default='',
        help='Path to output image folder. This is where the images from the run will be saved.'
    )
    args = parser.parse_args()

    # check that model Keras version is same as local Keras version
    f = h5py.File(args.model, mode='r')
    model_version = f.attrs.get('keras_version')
    keras_version = str(keras_version).encode('utf8')
    
    if model_version != keras_version:
        print('You are using Keras version ', keras_version,
              ', but the model was built using ', model_version)

    model = kn.KaNet(config.img_dim, config.resize_factor, config.output_size, config.output_activation, config.dropout, config.weight_decay, 0)
    model.load_weights(args.model)
    
    if args.image_folder != '':
        if not os.path.exists(args.image_folder):
            print("Creating output folder: {}".format(args.image_folder))
            os.makedirs(args.image_folder)
        else:
            shutil.rmtree(args.image_folder)
            os.makedirs(args.image_folder)
        print("RECORDING ...")
    else:
        print("NOT RECORDING ...")

    # wrap Flask application with engineio's middleware
    app = socketio.Middleware(sio, app)

    # deploy as an eventlet WSGI server
    eventlet.wsgi.server(eventlet.listen(('', 4567)), app)
