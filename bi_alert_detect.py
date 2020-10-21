"""
UPDATE 09 June 2020: Migrated to python 3.6.8

___
This module uses AI image recongition tools to determine whether or not specific
objects are present in a Blue Iris CCTV snapshot image. If objects are found, it tries
to determine if there has been movement since previous notifications, e.g. for parked cars.

Currently, it can use one of:
    i.      Google's TensorFlow prebuilt model (from the TensorFlow zoo), running locally
    ii.     Amazon's AWS Rekognition cloud based solution
    iii.    SightHound's cloud based solution

Image recogition is triggered by receiving a notification via MQTT containing the BlueIris camera name.
The module then downloads a snapshot for the camera, and pushes this through the selected AI framework.
Blue boxes are drawn around objects found on the image, that meet the object search criteria of the camera.
Yellow boxes are drawn around objects that appear stationary.
The processed image cane be saved in the designated folder for later review.

Once objects are detected, notifications from the module are sent back via:
    a.  MQTT (this can be picked up by BlueIris for further action)
    b.  Telegram
    c.  PushBullet

Additional libraries required include:

    Core libraries:
        -   cv2:            pip install opencv-python
        -   numpy:          pip install numpy
        -   paho:           pip install paho-mqtt

    Model framework libraries (depending on which model(s) are used):
        -   tensorflow:     pip install tensorflow
        -   Amazon AWS:     pip install install awscli --upgrade
                            pip install boto3

    Optional libraries, depending on notification method used (if any):
        -   telegram:       pip install python-telegram-bot --upgrade
        -   pusbullet:      https://github.com/rbrcsk/pushbullet.py

In addition, for TensorFlow, an appropriate prebuilt model from the TensorFlow zoo is required, e.g.
http://download.tensorflow.org/models/object_detection/faster_rcnn_inception_v2_coco_2018_01_28.tar.gz

Configuration options and parameters need to be specified in the file config.py

"""

# Note that model specific imports are in do_initialise_model() function
# and Telegram/Pushbullet imports in the __main__ section. This is to avoid unnecessary
# imports which may also take some time to load, e.g. tensorflow


import sys
import base64
import json
import ssl
import urllib.request, urllib.parse, urllib.error
from datetime import datetime
import time
import os
import signal
import http.client
from collections import namedtuple
from argparse import ArgumentParser
import pickle
import glob
import numpy as np
import tensorflow.compat.v1 as tf
import boto3
import cv2
# import logging

import config
import coco_labels as coco



# Globals ---------------------------------------------------------------------------------
DEBUG = config.DEBUG
VERSION = "2.0"
# logger = logging.getLogger('bi_detect')
# logger.setLevel(logging.INFO if not config.DEBUG else logging.DEBUG)

# ch = logging.StreamHandler(sys.stdout)
# ch.setFormatter(logging.Formatter("%(asctime)s - %(levelname)s - %(message)s"))
# logger.addHandler(ch)
# logger.propagate = False

DETECTION_DISABLED = False

mqtt_client = None

# Note that the ai_detector global var is only assigned in the do_initialise_model() fn
ai_detector = None

# Note that the last_prune_date variable is only changed in the delete_old_images() fn
last_prune_date = None


# Identified object boundary boxes
BOX_COLOUR_TRIGGER = (255, 0, 0)
TEXT_COLOUR_TRIGGER = (255, 255, 255)
BOX_THICKNESS_TRIGGER = 2
BOX_COLOUR_NON_TRIGGER = (0, 255, 255)
TEXT_COLOUR_NON_TRIGGER = (0, 0, 0)
BOX_THICKNESS_NON_TRIGGER = 1
TAG_FONT_SCALE = 0.4

# Minimum boundary box overlap with previous image, for a stationary object
STATIONARY_OBJECT_OVERLAP_THRESHOLD = 80.0
IGNORE_REGION_THRESHOLD = 80.0

# Previous alert filename (for detecting initial stationary objects on first run)
PREVIOUS_ALERTS_FILE = "prev_alerts.pickle"

# Virtual 'camera' for processing image files (i.e. not images downloaded from a BI camera)
IMAGE_CAMERA = "_IMAGE_FILE"

# Model names
TF = "TF"
SH = "SH"
AWS = "AWS"

TITLE = "BlueIris Enhanced Object Detection System"

# Boundary box corners
Rectangle = namedtuple('Rectangle', 'xmin ymin xmax ymax')

class IdentifiedObject(object):
    ''' Object identified in an image'''

    def __init__(self, label=None, confidence=None, bounding_box=None, is_stationary=False, in_ignore_zone=False):
        self.label = label
        self.confidence = confidence
        self.box = bounding_box
        self.is_stationary = is_stationary
        self.in_ignore_zone = in_ignore_zone

    def __repr__(self):
        return "IdentifiedObject({}, {}, {}, {})".format(
            self.label, self.confidence, self.box, self.is_stationary)

class ImageFrame(object):
    ''' Image snapshot object, containing details of the image, including list of
        objects identified,  alert triggerring objects and file names for saved images
    '''
    def __init__(self, camera_name, image_binary=None, filename=None, timestamp=time.time()):
        self.camera_name = camera_name
        self.timestamp = timestamp
        self.trigger_image = image_binary
        self.processed_image = None
        self.trigger_file = filename
        self.processed_file = None
        self.detection_model = None
        self._init_new_image()

    def _init_new_image(self):
        if self.trigger_image is not None:
            self.height, self.width, _ = self.trigger_image.shape
        else:
            self.width = None
            self.height = None
        self.detected_objects = []
        self.alert_objects = []

    def __repr__(self):
        return "ImageFrame(cameraname='{}', timestamp={}, trigger_file='{}', " \
                "processed_file='{}', detection_model='{}', height={}, width={})".format(
                    self.camera_name, self.timestamp, self.trigger_file, self.processed_file,
                    self.detection_model, self.height, self.width)

    # --------------------------------------------------
    def get_image_from_camera(self, url):
        """ Download the image, convert it to a NumPy array, and then read it into OpenCV format """
        if DEBUG:
            print("[DEBUG] Getting image from BlueIris url: %s" % url)

        resp = urllib.request.urlopen(url)
        image = np.asarray(bytearray(resp.read()), dtype="uint8")
        image = cv2.imdecode(image, cv2.IMREAD_UNCHANGED)
        self.timestamp = time.time()
        self.trigger_image = image
        self.processed_image = image    # Start off by having processed image same as initial image

        self._init_new_image()
        # if DEBUG:
        #     # print("[DEBUG] [ImageFrame.get_image_from_camera] Image width: {}, height: {}".format(
        #         self.width, self.height))

        # return the image
        return self.trigger_image

    # --------------------------------------------------
    def load_image_from_file(self, file_name):
        if DEBUG:
            print("[DEBUG] Loading image '%s' from disk" % file_name)
        self.trigger_image = cv2.imread(file_name, cv2.IMREAD_UNCHANGED)
        self.processed_image = self.trigger_image
        self._init_new_image()
        self.trigger_file = file_name
        self.timestamp = time.time()
        print("[DEBUG] [ImageFrame.load_image_from_file] Image width: {}, height: {}".format(
            self.width, self.height))
        # return the image
        return self.trigger_image

    # --------------------------------------------------
    def get_alert_objects_list(self, separator=", "):
        items = set([item.label for item in self.alert_objects])
        alert_list = separator.join(items)
        return alert_list

    def save_trigger_image(self, filename=None):
        """ Save the original image that triggered a detection process (regardless of whether or not object detected). """
        if filename:
            self.trigger_file = filename

        if not self.trigger_file:
            self.trigger_file = os.path.join(
                config.IMAGE_SAVE_PATH, "%s_%s.jpg" % (
                    time.strftime("%Y-%m-%d_%H.%M.%S", time.localtime(self.timestamp)),
                    self.camera_name))
        ret = cv2.imwrite(self.trigger_file, self.trigger_image)
        return ret

    # --------------------------------------------------
    def save_processed_image(self, filename=None):
        """ Save the processed image, showing object bounding boxes, if any. """
        if filename is None:
            tag = self.get_alert_objects_list("_")
            filename = os.path.join(
                config.IMAGE_SAVE_PATH, "%s_%s_%s_%s.jpg" %
                (time.strftime("%Y-%m-%d_%H.%M.%S", time.localtime(self.timestamp)),
                 self.camera_name, self.detection_model, tag))
            # print("Using filename %s" % filename)
        ret = cv2.imwrite(filename, self.processed_image)
        self.processed_file = filename
        return ret

class Camera(object):
    ''' Object holding details of Blue Iris cctv camera    '''

    def __init__(self, camera_name, user_id=config.BI_USER,
                 password=config.BI_PW, protocol="http",
                 server=None, port=None,
                 quality=config.BI_IMAGE_QUALITY, scale=config.BI_IMAGE_SCALE):
        self.name = camera_name
        self.user_id = user_id
        self.password = password
        self.protocol = protocol
        self.server = server
        self.port = port
        self.image_quality = quality
        self.image_scale = scale
        self.search_objects = self.get_search_items()
        self.ignore_regions = self.get_ignore_regions()
        self.url = self.get_url()
        self.alert = None
        self.previous_objects = []

    def get_search_items(self):
        """ Get list of object types (from config file, if defined there, otherwise using defaults) that we should
        be looking for, for this camera.
        """
        if hasattr(config, "DETECT_OBJECTS_BY_CAMERA") and self.name in config.DETECT_OBJECTS_BY_CAMERA:
            items = config.DETECT_OBJECTS_BY_CAMERA[self.name]
        else:
            items = config.DETECT_OBJECTS_DEFAULT
        if DEBUG:
            print("[DEBUG] Search objects for camera '{}': {}".format(self.name, items))
        return items

    def get_previous_objects_list(self, separator=", "):
        items = set([item.label for item in self.previous_objects])
        string_list = separator.join(items)
        return string_list

    def get_ignore_regions(self):
        if hasattr(config, "IGNORE_CAM_REGIONS") and self.name in config.IGNORE_CAM_REGIONS:
            regions = config.IGNORE_CAM_REGIONS[self.name]
        else:
            regions = []
        return regions

    def get_url(self):
        if self.server:
            self.url = self.protocol + "://" + self.server
            if self.port:
                self.url = "".join((self.url, ":", str(self.port)))

            self.url = "".join((self.url, "/image/", self.name))

            if self.user_id:
                self.url = "".join((self.url, "?user=", self.user_id))
                if self.password:
                    self.url = "".join((self.url, "&pw=", self.password))

            self.url = "".join((self.url, "&q=", str(self.image_quality), "&s=", str(self.image_scale)))
        else:
            self.url = ""

        return self.url

    def get_trigger_image(self, from_image_file=""):
        ''' Downloads a snapshot image from BlueIris (for the current camera), which is then to be
            examined for any of the required search objects (this is referred to as the "trigger" image '''
        self.alert = ImageFrame(self.name)
        if from_image_file and os.path.isfile(from_image_file):
            self.alert.load_image_from_file(from_image_file)
            return self.alert.trigger_image
        else:
            if self.url is None:
                self.url = self.get_url()
             
            if self.url is not None and self.url:
                self.alert.get_image_from_camera(self.url)
                return self.alert.trigger_image
            else:
                print("[INFO ][get_trigger_image] Camera image download url not defined")

    def load_trigger_image_file(self, image_file_name):
        ''' Read image from file '''

        # img = cv2.imread('./input/%s.jpg' % count, cv2.IMREAD_UNCHANGED)
        # img = cv2.resize(img, (1280, 720)) #Blue Iris image at 0.8 quality should be 1280x720 already

    def rectangle_overlap_percentage(self, a, b):
        ''' Calculates % overlap of box b on box a, if any   '''

        dx = min(a.xmax, b.xmax) - max(a.xmin, b.xmin)
        dy = min(a.ymax, b.ymax) - max(a.ymin, b.ymin)
        # print("dx: {}, dy: {}".format(dx, dy))
        if (dx >= 0) and (dy >= 0):
            area_intersect = float(dx * dy)
            area_box1 = float((a.xmax - a.xmin) * (a.ymax - a.ymin))
            area_box2 = float((b.xmax - b.xmin) * (b.ymax - b.ymin))
            area_union = area_box1 + area_box2  - area_intersect
            overlap = area_intersect/area_union * 100
        else:
            overlap = 0.0
        return overlap

    def is_stationary(self, idx, label, pos):
        ''' Function to determine whether an object is approx stationary as compared to previous image
            from same camera '''

        is_stationary = False
        idx += 1 # Aesthetics, as zero based
        if self.previous_objects:
            # Check if the objects are in roughly the same place
            for obj in self.previous_objects:
                prev_pos = obj.box
                overlap_percentage = self.rectangle_overlap_percentage(pos, prev_pos)
                if overlap_percentage > STATIONARY_OBJECT_OVERLAP_THRESHOLD:
                    # Object matched with previous location data, and is in approx same location
                    if DEBUG:
                        print("[DEBUG] [is_stationary] {} {}: Appears stationary ({:.2f}%) against " \
                            "prev object at ({},{})/({},{})".format(
                                obj.label, idx, overlap_percentage, prev_pos.xmin, prev_pos.ymin,
                                prev_pos.xmax, prev_pos.ymax))
                    is_stationary = True
                    break
                elif DEBUG:
                    print("[DEBUG] [is_stationary] {} {}: No match ({:.2f}% overlap) against " \
                        "prev object at ({},{})/({},{})".format(
                            obj.label, idx, overlap_percentage, prev_pos.xmin, prev_pos.ymin,
                            prev_pos.xmax, prev_pos.ymax))
        else:
            if DEBUG:
                print("[DEBUG] [is_stationary] Object {} {}: Ignoring stationary check as no "\
                    "previous object positions available".format(label, idx))

        return is_stationary

    def is_in_ignore_region(self, box, label):
        ''' Function to determine whether an object box is in an "ignore" region for the camera '''
        if hasattr(config, "IGNORE_CAM_REGIONS") and self.name in config.IGNORE_CAM_REGIONS:
            ignore_regions = config.IGNORE_CAM_REGIONS[self.name]
            for region in ignore_regions:
                i_xmin = region[0][0]
                i_xmax = region[1][0]
                i_ymin = region[0][1]
                i_ymax = region[1][1]

                # Check overlap
                dx = min(i_xmax, box.xmax) - max(i_xmin, box.xmin)
                dy = min(i_ymax, box.ymax) - max(i_ymin, box.ymin)

                if (dx >= 0) and (dy >= 0):
                    area_intersect = float(dx * dy)
                    area_org_box = float((box.xmax - box.xmin) * (box.ymax - box.ymin))
                    percent_overlap = area_intersect/area_org_box * 100
                    if DEBUG:
                        print("[DEBUG] [ignore_region] {}: {:.2f}% in ignore region ({},{})/({},{})"\
                            .format(label, percent_overlap, i_xmin, i_ymin, i_xmax, i_ymax))

                # if x_in_ignore_region and y_in_ignore_region:
                    if percent_overlap >= IGNORE_REGION_THRESHOLD:
                        return True
        return False

class AIObjectDetector(object):
    ''' Main AI detector object for the different AI frameworks '''

    def __init__(self, framework=None, image=None):
        self.SIGHTHOUND_HEADERS = {"Content-type": "application/json",
                                       "X-Access-Token": config.SIGHTHOUND_CLOUD_TOKEN}

        self.framework = framework
        self.framework_folder = None
        self.image = image
        self.cameras = {}
        self.active_cam = None

        self.tf_api = None
        self.aws_client = None
        if self.framework is not None:
            self.init_framework()
        self.load_previous_alerts()


    def init_framework(self, model_name=None):
        if model_name is not None:
            self.framework = model_name
        if self.framework == TF and self.tf_api is None:
            self.framework_folder = config.TENSORFLOW_MODEL_DIRECTORY
            self.framework_graph_path = os.path.join(config.TENSORFLOW_MODEL_DIRECTORY, config.FROZEN_INFERENCE_GRAPH)
            print("[INFO ] Loading model from folder '%s' (may take some time)..." % self.framework_folder)
            self.tf_api = self.TFDetectorAPI(path_to_ckpt=self.framework_graph_path)
        elif self.framework == AWS and self.aws_client is None:
            print("[INFO ] Initialising AWS Rekognition (boto3) client")
            self.aws_client = boto3.client('rekognition')
        else:
            if DEBUG:
                print("[DEBUG] Model {} is already initalised. Nothing further to do".format(self.framework))

    def save_previous_alerts(self):
        if self.cameras:
            with open(PREVIOUS_ALERTS_FILE, "wb") as handle:
                pickle.dump(self.cameras, handle, protocol=pickle.HIGHEST_PROTOCOL)
        else:
            if DEBUG:
                print("[DEBUG] Ignoring save history as no cameras collection available")


    def load_previous_alerts(self):
        if DEBUG:
            print("[DEBUG] Loading previous stationary objects history")
        if os.path.isfile(PREVIOUS_ALERTS_FILE):
            with open(PREVIOUS_ALERTS_FILE, "rb") as handle:
                self.cameras = pickle.load(handle)
                # Reset camera urls, in case of changes in config
                for cam in self.cameras: 
                    self.cameras[cam].url = None
                    self.cameras[cam].image_quality = config.BI_IMAGE_QUALITY
                    self.cameras[cam].image_scale = config.BI_IMAGE_SCALE


    # --------------------------------------------------
    def normalised_box_to_coordinates(self, box, image=None, im_width=None, im_height=None):
        if image is not None:
            im_height, im_width, _ = image.shape
        x = int(box["Left"] * im_width)
        y = int(box["Top"] * im_height)
        x1 = int(box["Width"] * im_width) + x
        y1 = int(box["Height"] * im_height) + y
        return x, y, x1, y1


    # --------------------------------------------------
    def normalised_coordinates_to_box(self, left, top, width, height, image=None, im_width=None, im_height=None):
        if image is not None:
            im_height, im_width, _ = image.shape
        x = int(left * im_width)
        y = int(top * im_height)
        x1 = int(width * im_width) + x
        y1 = int(height * im_height) + y
        box = Rectangle(x, y, x1, y1)
        return box


    # --------------------------------------------------
    def normalised_box_to_box(self, box, image=None, im_width=None, im_height=None):
        x, y, x1, y1 = self.normalised_box_to_coordinates(box, image, im_width, im_height)
        return Rectangle(x, y, x1, y1)


    # --------------------------------------------------
    def draw_bounding_box(self, obj):
        # if obj.label in self.active_cam.search_objects:

        if obj.is_stationary:
            box_colour = BOX_COLOUR_NON_TRIGGER
            text_colour = TEXT_COLOUR_NON_TRIGGER
            line_thickness = BOX_THICKNESS_NON_TRIGGER
        else:
            box_colour = BOX_COLOUR_TRIGGER
            text_colour = TEXT_COLOUR_TRIGGER
            line_thickness = BOX_THICKNESS_TRIGGER

        # Box the object...
        cv2.rectangle(self.active_cam.alert.processed_image, (obj.box.xmin, obj.box.ymin),
                      (obj.box.xmax, obj.box.ymax), box_colour, line_thickness)

        # Tag it ...
        if config.TAG_IMAGES:
            tag = obj.label #(obj.label + " " + str(instance_count)) if instance_count > 1 else obj.label
            text_width, text_height = cv2.getTextSize(
                tag, cv2.FONT_HERSHEY_SIMPLEX, fontScale=TAG_FONT_SCALE, thickness=1)[0]
            cv2.rectangle(
                self.active_cam.alert.processed_image,
                (obj.box.xmin, obj.box.ymin),
                (obj.box.xmin + text_width + 10, obj.box.ymin - text_height - 5),
                box_colour, -1)                        
            cv2.putText(
                self.active_cam.alert.processed_image, tag,
                (obj.box.xmin + 2, obj.box.ymin - int(text_height / 2)),
                cv2.FONT_HERSHEY_SIMPLEX, TAG_FONT_SCALE, text_colour, 1)


    # --------------------------------------------------
    def do_detect(self, cam_name, local_image_file=""):
        if DEBUG:
            print("[DEBUG] 'do_detect' function called for {}".format(
            "camera {}".format(cam_name) if not local_image_file else "image file '{}'".format(local_image_file)))

        if local_image_file:
            cam_name = os.path.basename(local_image_file)

        if cam_name in self.cameras:
            if DEBUG:
                print("[DEBUG] Camera found in existing collection")
            self.active_cam = self.cameras[cam_name]
            if DEBUG:
                print("[DEBUG] Previous stationary objects list: " + self.active_cam.get_previous_objects_list())
        else:
            if DEBUG:
                print("[DEBUG] Camera '%s' not present in existing collection. " \
                    "Instantinating new instance" % cam_name)
            self.cameras[cam_name] = Camera(
                cam_name, config.BI_USER, config.BI_PW,
                config.BI_PROTOCOL, config.BI_SERVER, config.BI_PORT)
            if DEBUG:
                print("[DEBUG] New camera '{}' added to dictionary".format(cam_name))
            self.active_cam = self.cameras[cam_name]

        if DEBUG:
            if not local_image_file:
                print("[DEBUG] Downloadinge image from BlueIris")
            else:
                print("[DEBUG] Loading image from file '%s'" %local_image_file)
        download_start_time = time.time()

        if local_image_file:
            self.active_cam.get_trigger_image(local_image_file)
        else:
            self.active_cam.get_trigger_image()
        self.active_cam.alert.detection_model = self.framework

        # img = cv2.resize(img, (1280, 720)) #Blue Iris image at 0.8 quality should be 1280x720 already

        print("[INFO ] Image loaded/downloaded from BI ({:.2f} seconds) [{} x {}]".format(
            time.time() - download_start_time, self.active_cam.alert.width, self.active_cam.alert.height))

        if self.framework == TF:
            if DEBUG:
                print("[DEBUG] Calling 'process_tensorflow'")
            self.process_tensorflow()

        elif self.framework == SH:
            if DEBUG:
                print("[DEBUG] Calling 'process_sighthound'")
            self.process_sighthound()

        elif self.framework == AWS:
            if DEBUG:
                print("[DEBUG] Calling 'process_aws'")
            self.process_aws()

        else:
            print("Invalid model system. Aborting detection...")
            return False

        if DEBUG:
            print("[DEBUG] Returned from detection API function. Processing identified objects")

        if self.active_cam.alert.detected_objects:
            # Check if detected objects (which are already above required confidence level and in watch list)
            # have bounding boxes and - for non-Person objects - are moving/stationary
            for idx, obj in enumerate(self.active_cam.alert.detected_objects):
                obj.in_ignore_zone = self.active_cam.is_in_ignore_region(obj.box, obj.label + " " + str(idx + 1))
                if not obj.in_ignore_zone:
                    obj.is_stationary = self.active_cam.is_stationary(idx, obj.label, obj.box)
                    if not obj.is_stationary:
                        self.active_cam.alert.alert_objects.append(obj)
                    # Draw box - this is after checking for stationary, so we get the right colours
                    self.draw_bounding_box(obj)
                else:
                    if DEBUG:
                        print("[DEBUG] Ignoring '{} {}' as in ignore region (object position: [{},{})/({},{})]"\
                            .format(obj.label, idx + 1, obj.box.xmin, obj.box.ymin, obj.box.xmax, obj.box.ymax))

        if self.active_cam.alert.alert_objects:
            if config.SAVE_PROCESSED_IMAGE:
                self.active_cam.alert.save_processed_image()
        else:
            if config.SAVE_ORIGINAL_IMAGE:
                self.active_cam.alert.save_trigger_image()

        self.active_cam.previous_objects = self.active_cam.alert.detected_objects

        return len(self.active_cam.alert.alert_objects) > 0

    # --------------------------------------------------
    def process_aws(self):
        if DEBUG:
            print("[DEBUG] AWS Rekognition: Classifying frame contents from camera '{}'".format(self.active_cam.name))

        # aws_client = boto3.client('rekognition')
        # with open("test.jpg", 'rb') as image:
        #     print("[DEBUG] TEST Posting image to AWS Rekognition API")
        #     response = aws_client.detect_labels(Image={'Bytes': image.read()})
        # print("done testing reponse")
        # print ("response... {}".format(response))
        timestamp = time.time()
        _, image_buffer = cv2.imencode('.jpg', self.active_cam.alert.trigger_image)

        response = self.aws_client.detect_labels(
            Image={'Bytes': image_buffer.tobytes()}, MinConfidence=config.CONFIDENCE_THRESHOLD)

        print("[INFO ] AWS Rekognition response ({:.2f} seconds):".format(time.time() - timestamp))

        self.active_cam.alert.detected_objects = []
        for label in response['Labels']:
            if label["Name"] in self.active_cam.search_objects and label["Instances"]:
                for instance in label["Instances"]:
                    box = self.normalised_box_to_box(
                        instance["BoundingBox"], None, self.active_cam.alert.width, self.active_cam.alert.height)
                    obj = IdentifiedObject(label["Name"], label['Confidence'], box)
                    self.active_cam.alert.detected_objects.append(obj)
                    # print("[DEBUG][AWS] -> {:<20}: {:.2f}% (Bounding box: [{}, {}], [{}, {}])".format(
                    #   label['Name'], label['Confidence'], box.xmin, box.ymin, box.xmax, box.ymax))
            else:
                if DEBUG:
                    print("[DEBUG] -> {:<20}: {:.2f}% [ignored - not in search list]".format(label['Name'], label['Confidence']))

        if DEBUG:
            print("[DEBUG] Completed AWS Recognition")


    # --------------------------------------------------
    def process_sighthound(self):
        # Note SightHound API checks for only people OR cars in single call (not both).
        # Only consider people for now to save API calls

        if DEBUG:
            print("[DEBUG] Converting image to base64")
        _, image_buffer = cv2.imencode('.jpg', self.active_cam.alert.trigger_image)
        image_data = base64.b64encode(image_buffer).decode()

        timestamp = time.time()

        # DEBUG.............
        # print("opening test.jpt")
        # image_file_name = "test.jpg"
        # image = open(image_file_name, "rb").read()
        # print("test.jpg opened")
        # image_data = base64.b64encode(image).decode()

        params = json.dumps({"image": image_data})

        print("[DEBUG] Posting image to SightHound API")
        sighthound_connection = http.client.HTTPSConnection("dev.sighthoundapi.com",
                                                        context=ssl.SSLContext(ssl.PROTOCOL_TLSv1))
        sighthound_connection.request("POST", "/v1/detections?type=face, person&faceOption=landmark, gender",
                                      params, self.SIGHTHOUND_HEADERS)
        response = sighthound_connection.getresponse()
        result = response.read()
        print("[DEBUG] Sighthound response: {} ({:.2f} seconds)".format(result, time.time() - timestamp))

        if "person" in result:
            count = 0
            modified_result = result.replace('"x"', '"Left"').replace('"y"', '"Top"')\
                .replace('"height"', '"Height"').replace('"width"', '"Width"').replace('"person"', '"Person"')
            response = json.loads(modified_result)
            for o in response["objects"]:
                if o["type"] == "Person":
                    count += 1
                    if o["boundingBox"]:
                        box = self.normalised_box_to_box(
                            o["boundingBox"], None, response["image"]["Width"], response["image"]["Height"])

                        # No confidence value returned by SH. Defaulting to our threshold value
                        obj = IdentifiedObject("Person", config.CONFIDENCE_THRESHOLD, box)

                        self.active_cam.alert.detected_objects.append(obj)
                        # print("[DEBUG] -> {:<20} (Bounding box: [{}, {}], [{}, {}])"\
                        # .format("Person", box.xmin, box.ymin, box.xmax, box.ymax))
                    else:
                        print("[INFO ] -> {:<20}".format(o["type"]))

        if DEBUG:
            print("[DEBUG] Completed SightHound processing")


    # --------------------------------------------------
    def process_tensorflow(self):
        if DEBUG:
            print("[DEBUG] TensorFlow: Classifying frame contents from camera '{}', with model '{}'".format(
                self.active_cam.name, self.framework_folder))
        boxes, scores, classes, num = self.tf_api.process_frame(self.active_cam.alert.trigger_image)
        self.active_cam.alert.detected_objects = []
        if DEBUG:
            print("[DEBUG] Reviewing classfications for {} detected objects".format(num))

        for i, _ in enumerate(boxes):
            confidence_level = scores[i] * 100
            if len(coco.LABELS) >= classes[i]:
                label = coco.LABELS[classes[i]]

                # Check if the classfications found are in our wanted list
                if label in self.active_cam.search_objects and confidence_level >= config.CONFIDENCE_THRESHOLD:
                    obj = IdentifiedObject(label, confidence_level, boxes[i])
                    self.active_cam.alert.detected_objects.append(obj)
                elif DEBUG and confidence_level >= config.CONFIDENCE_THRESHOLD:
                    print("[DEBUG] -> {:<20} {:.2f}% [ignored - not in search list]"\
                        .format(coco.LABELS[classes[i]], confidence_level))
            elif DEBUG:
                print("[DEBUG] -> Unknown object (Label ID {:<20}): {:.2f}% [droppped]"\
                    .format(classes[i], confidence_level))
        if DEBUG:
            print("[DEBUG] Completed TF processing. Returning to calling function")


    # TensorFlow class object --------------------------------
    class TFDetectorAPI(object):
        ''' TensorFlow API '''

        def __init__(self, path_to_ckpt):
            self.path_to_ckpt = path_to_ckpt

            self.detection_graph = tf.Graph()
            with self.detection_graph.as_default():
                od_graph_def = tf.GraphDef()
                with tf.gfile.GFile(self.path_to_ckpt, 'rb') as fid:
                    serialized_graph = fid.read()
                    od_graph_def.ParseFromString(serialized_graph)
                    tf.import_graph_def(od_graph_def, name='')

            self.default_graph = self.detection_graph.as_default()
            self.sess = tf.Session(graph=self.detection_graph)

            # Definite input and output Tensors for detection_graph
            self.image_tensor = self.detection_graph.get_tensor_by_name('image_tensor:0')
            # Each box represents a part of the image where a particular object was detected.
            self.detection_boxes = self.detection_graph.get_tensor_by_name('detection_boxes:0')
            # Each score represent how level of confidence for each of the objects.
            # Score is shown on the result image, together with the class label.
            self.detection_scores = self.detection_graph.get_tensor_by_name('detection_scores:0')
            self.detection_classes = self.detection_graph.get_tensor_by_name('detection_classes:0')
            self.num_detections = self.detection_graph.get_tensor_by_name('num_detections:0')

        def process_frame(self, image):
            # Expand dimensions since the trained_model expects images to have shape: [1, None, None, 3]
            image_np_expanded = np.expand_dims(image, axis=0)
            # Actual detection
            frame_start_time = time.time()
            (boxes, scores, classes, num) = self.sess.run(
                [self.detection_boxes, self.detection_scores, self.detection_classes, self.num_detections],
                feed_dict={self.image_tensor: image_np_expanded})
            frame_end_time = time.time()

            if DEBUG:
                print("[DEBUG] Frame content classification took {0:.4f} seconds ".format(
                    frame_end_time - frame_start_time))

            im_height, im_width, _ = image.shape
            boxes_list = [None for i in range(boxes.shape[1])]
            for i in range(boxes.shape[1]):
                boxes_list[i] = Rectangle(
                    int(boxes[0, i, 1] * im_width),
                    int(boxes[0, i, 0] * im_height),
                    int(boxes[0, i, 3] * im_width),
                    int(boxes[0, i, 2] * im_height))
            return boxes_list, scores[0].tolist(), [int(x) for x in classes[0].tolist()], int(num[0])

        # def close(self):
        #     self.sess.close()
        #     self.default_graph.close()


# Functions ======================================================================================================
def do_initialise_model(model_name):
    ''' Initialise the selected AI framework. This is done in a function
        as model may be changed whilst script is running
    '''
    global ai_detector

    if model_name == AWS:
        print("[INFO ] Initialising AWS Rekognition SDK")
        initalised = True

    elif model_name == SH:
        print("[INFO ] Initialising SightHound system")
        initalised = True

    elif model_name == TF:
        full_path = os.path.join(config.TENSORFLOW_MODEL_DIRECTORY, config.FROZEN_INFERENCE_GRAPH)
        if not (os.path.isdir(config.TENSORFLOW_MODEL_DIRECTORY) and os.path.isfile(full_path)):
            print("[ERROR] Model inference graph file not found: '%s'" % full_path)
            sys.exit(0)
        # print("Initialising TensorFlow (may take a few seconds)")
        initalised = True
    else:
        initalised = False
    if initalised:
        if not ai_detector:
            ai_detector = AIObjectDetector(model_name)
        else:
            ai_detector.init_framework(model_name)
        print("[INFO ] '{}' model initalised".format(model_name))
    return initalised


def process_remote_command(message, source):
    global DEBUG
    global DETECTION_DISABLED

    print(" ")
    print("______________________________________________________________")
    print(time.strftime("%Y-%m-%d %H:%M:%S") + " '" + message + "' message received [" + source + "]")

    if message and "=" in message:
        command, arg = message.split("=", 1)
        command = command.lower().strip()
        if command == "camera":
            if DEBUG:
                print("[DEBUG] Alert received for camera '{}'".format(arg))
            do_recognition_for_camera_image(arg)
        elif command == "image":
            print("image...")
            if not os.path.isfile(arg):
                print("[ERROR] The image file '%s' was not found" % arg)
            else:
                if DEBUG:
                    print("[DEBUG] Detect requsted for image file '{}'".format(arg))
                do_recognition_for_camera_image(IMAGE_CAMERA, arg)
        elif command == "debug":
            DEBUG = (arg.lower() == "true")
            print("[INFO ] Debug mode set to '{}'".format(DEBUG))
        elif command == "model" and arg.upper() in [TF, AWS, SH]:
            do_initialise_model(arg.upper())
        elif command == "enabled":
            DETECTION_DISABLED = (arg.lower() == "false")
            if DETECTION_DISABLED:
                print("[INFO ] Image recognition has been disabled")
            else:
                print("[INFO ] Image recognition re-enabled")
        else:
            print("[INFO ] The command {} is not supported".format(message))
    else:
        # Temporary -  Falling back to assuming  the message is the camera name as before. TODO...!!!
        do_recognition_for_camera_image(message)


def do_recognition_for_camera_image(full_cam_name, local_image_file=None):
    ''' Main function to manage the overall recognition process '''

    if DETECTION_DISABLED:
        if DEBUG:
            print("[DEBUG] Ignoring alert as image recognition mode is disabled")
        return

    start_time = time.time()

    if config.BI_CLONED_CAMERA_COMMON_NAMING:
        cam_name = full_cam_name.split("_", 1)[0]
        if DEBUG and full_cam_name.contains("_"):
            print("[DEBUG] Camera '{}' identified as clone of '{}'".format(full_cam_name, cam_name))
    else:
        cam_name = full_cam_name

    if hasattr(config, "MINUTES_BETWEEN_SAME_CAMERA_ALERTS") and config.MINUTES_BETWEEN_SAME_CAMERA_ALERTS > 0:
        if cam_name in ai_detector.cameras and ai_detector.cameras[cam_name].alert and ai_detector.cameras[cam_name].alert.alert_objects:
            seconds_since_last_alert = time.time() - ai_detector.cameras[cam_name].alert.timestamp
            if seconds_since_last_alert < (config.MINUTES_BETWEEN_SAME_CAMERA_ALERTS * 60):
                print("[INFO ] Dropping BI alert notification, as still within the timeout period " \
                    "of the last alert ({:.0f} seconds ago)".format(seconds_since_last_alert))
                return
    if DEBUG:
        print("[DEBUG] Calling image processing function...")
    detected = ai_detector.do_detect(cam_name, local_image_file)
    if detected:
        for idx, obj in enumerate(ai_detector.active_cam.alert.detected_objects):
            if obj.in_ignore_zone:
                label = obj.label + " " + str(idx + 1) + " [ignore zone]"
            else:
                label = (obj.label + " " + str(idx + 1) + " [stationary]") if obj.is_stationary else obj.label + " " + str(idx + 1)

            print("[INFO ] -> {:<20}: {:.2f}% (Bounding box: [{}, {}], [{}, {}])"\
                .format(label, obj.confidence, obj.box.xmin, obj.box.ymin, obj.box.xmax, obj.box.ymax))

        detected_items = ai_detector.active_cam.alert.get_alert_objects_list(", ")
        print("[INFO ] Motion/objects detected: " + detected_items + " ({0:.2f} seconds overall)"\
            .format(time.time() - start_time))

        if send_notification(ai_detector.active_cam.name, ai_detector.active_cam.alert.processed_file):
            print("------> Notification sent")
        ai_detector.save_previous_alerts()
    else:
        print("[INFO ] Analysis complete. No motion detected ({0:.2f} seconds overall)"\
            .format(time.time() - start_time))

    # Check and prune any old images
    delete_old_images()


def signal_handler(sig, frame):
    if mqtt_client:
        mqtt_client.loop_stop()
        mqtt_client.disconnect()
    print(TITLE + ' terminated')
    sys.exit(0)


def send_notification(cam_name, image_file_name):
    ''' Send Notifications '''
    if DEBUG:
        print("[DEBUG] send_notification called for camera '%s' and image file '%s'" % (cam_name, image_file_name))

    notification_sent = False
    if config.PUSH_MQTT_ALERT:
        if DEBUG:
            print("[DEBUG] Sending MQTT notification")
        mqtt_client.publish(config.MQTT_PUBLISH_TOPIC, "camera=" + cam_name + "_alert&trigger")
        if DEBUG:
            print("[DEBUG] mqtt message posted: " + "camera=" + cam_name + "_alert&trigger")
        notification_sent = True

    if pushbullet_notify:
        if DEBUG:
            print("[DEBUG] Sending PusBullet notification")
        with open(image_file_name, "rb") as pic:
            file_data = pushbullet_notify.upload_file(pic, os.path.basename(image_file_name))
        pushbullet_notify.push_file(**file_data)
        if DEBUG:
            print("[DEBUG] PushBullet notification sent")
        notification_sent = True

    if telegram_notify:
        if DEBUG:
            print("[DEBUG] Sending telegram notification with image file '%s'" % image_file_name)

        if image_file_name:
            telegram_notify.send_photo(
                chat_id=config.TELEGRAM_CHAT_ID,
                photo=open(image_file_name, 'rb'),
                parse_mode="Markdown",
                caption="Motion detected on [{}]({}://{}:{}@{}:{}/mjpg/{}/video.mjpg) camera".format(cam_name, config.BI_PROTOCOL,
                    config.BI_USER, config.BI_PW, config.BI_SERVER, config.BI_PORT, cam_name))

            if DEBUG:
                print("[DEBUG] Telegram notification sent")
        else:
            print("[DEBUG] Telegram message not sent as image from file '{}' not found".format(image_file_name))
        notification_sent = True

    return notification_sent


def delete_old_images():
    global last_prune_date

    if last_prune_date is None or last_prune_date < datetime.today().date():
        now = time.time()
        cutoff = now - config.DAYS_TO_KEEP_SAVED_IMAGES * 24 * 60 * 60
        count = 0
        for f in glob.glob(os.path.join(config.IMAGE_SAVE_PATH, "*.jpg")):
            if os.path.isfile(f):
                stat = os.stat(f)
                if stat.st_ctime < cutoff:
                    os.remove(f)
                    count += 1
        if DEBUG and count > 0:
            print ("[DEBUG] {} Old image files removed".format(count))
        last_prune_date = datetime.today().date()


# MQTT Functions ---------------------------------
def mqtt_on_connect(client, userdata, flags, rc):
    ''' mqtt connection event processing '''

    if rc == 0:
        client.connected_flag = True #set flag
        print("[INFO ] MQTT connection established with broker")
    else:
        print("[INFO ] MQTT connection failed (code {})".format(rc))
        if DEBUG:
            print("[DEBUG] mqtt userdata: {}, flags: {}, client: {}".format(userdata, flags, client))


def mqtt_on_disconnect(client, userdata, rc):
    ''' mqtt disconnection event processing '''

    client.loop_stop()
    if rc != 0:
        print("Unexpected disconnection.")
        if DEBUG:
            print("[DEBUG] mqtt rc: {}, userdata: {}, client: {}".format(rc, userdata, client))


def mqtt_on_log(client, obj, level, string):
    ''' mqtt log event received '''

    if DEBUG:
        print("[DEBUG] MQTT log message received. Client: {}, obj: {}, level: {}".format(client, obj, level))
    print("[DEBUG] MQTT log msg: {}".format(string))


def mqtt_on_message(client, userdata, msg):
    ''' mqtt message received on subscribed topic '''

    message = str(msg.payload, "utf-8")
    process_remote_command(message, "mqtt")


#===============================================================================================================

if __name__ == "__main__":

    signal.signal(signal.SIGINT, signal_handler)

    parser = ArgumentParser(description="Detect objects in CCTV image using AI tools")
    parser.add_argument("-i", "--image", dest="image_file", help="Load image from file", metavar="FILE")
    parser.add_argument("-d", "--debug", action="store_true", dest="debug", default=False,help="Print debug messages")

    args = parser.parse_args()
    single_image_file = None

    if args.debug is not None:
        DEBUG = True
    if args.image_file is not None:
        single_image_file = args.image_file
        if not os.path.isfile(single_image_file):
            print("[ERROR] The image file '%s' was not found" % single_image_file)
            sys.exit(2)


    print("")
    print("________________________________________________________________________________")
    print(TITLE + " started")
    print("[INFO ] Detection Framework: " + config.MODEL_SYSTEM)


    # Change to script folder
    abspath = os.path.abspath(__file__)
    dname = os.path.dirname(abspath)
    os.chdir(dname)

    # Instantiate main detector class (global)
    done_init_framework = do_initialise_model(config.MODEL_SYSTEM)

    if not done_init_framework:
        print("[DEBUG] Invalid modelling system. Please check the value in 'config.py' file")
        sys.exit(0)

    if hasattr(config, "PUSH_PUSHBULLET_NOTIFICATION") and hasattr(config, "PUSHBULLET_API_KEY") \
            and config.PUSH_PUSHBULLET_NOTIFICATION and config.PUSHBULLET_API_KEY:
        from pushbullet import Pushbullet
        pushbullet_notify = Pushbullet(config.PUSHBULLET_API_KEY)
        if DEBUG:
            print("[DEBUG] Pushbullet API initalised")
    else:
        pushbullet_notify = None

    if hasattr(config, "PUSH_TELEGRAM_ALERT") and hasattr(config, "TELEGRAM_API_TOKEN") \
            and hasattr(config, "TELEGRAM_CHAT_ID") and config.PUSH_TELEGRAM_ALERT \
            and config.TELEGRAM_API_TOKEN:
        import telegram
        telegram_notify = telegram.Bot(token=config.TELEGRAM_API_TOKEN)
        if DEBUG:
            print("[DEBUG] Telegram API initalised")
    else:
        telegram_notify = None

    if single_image_file:
        print("[INFO ] Processing image file '{}'".format(single_image_file))
        do_recognition_for_camera_image(IMAGE_CAMERA, single_image_file)
        print("Image '%s' processing complete" % single_image_file)
        sys.exit(0)

    # Prune old images on start up (only if we are not processing single image)
    delete_old_images()

    # Set up MQTT subscriber - TODO!!! Error trapping....
    if config.SUBSCRIBE_MQTT or config.PUSH_MQTT_ALERT:
        import paho.mqtt.client as mqtt

        mqtt_client = mqtt.Client()
        mqtt_client.username_pw_set(config.MQTT_USER, config.MQTT_PW)
        mqtt_client.on_connect = mqtt_on_connect
        mqtt_client.on_message = mqtt_on_message
        # mqtt_client.on_log = mqtt_on_log
        mqtt_client.on_disconnect = mqtt_on_disconnect

        print("[INFO ] Connecting to mqtt server %s" % config.MQTT_SERVER_NAME)
        mqtt_client.connect(config.MQTT_SERVER_NAME, port=1883, keepalive=0, bind_address="")

        if config.SUBSCRIBE_MQTT:
            print("[INFO ] Subscribing to mqtt topic '%s'" % config.MQTT_LISTEN_TOPIC)
            mqtt_client.subscribe(config.MQTT_LISTEN_TOPIC)

            mqtt_client.loop_forever()
            # client.loop_start() #start loop to process received messages

    elif config.BUILT_IN_WEB_SERVER:
        import http.server
        import socketserver
        print ("[INFO ] Instantiating built-in web server")
        class WebServerHandler(http.server.SimpleHTTPRequestHandler):
            def do_POST(self):
                content_len = int(self.headers.getheader('content-length', 0))
                post_body = self.rfile.read(content_len)
                process_remote_command(post_body, "http")

        web_server_handler = WebServerHandler
        httpd = socketserver.TCPServer(("", config.BUILT_IN_WEB_SERVER_PORT), web_server_handler)
        print("[INFO ] Webserver listening on port %s" % str(config.BUILT_IN_WEB_SERVER_PORT))
        httpd.serve_forever()
