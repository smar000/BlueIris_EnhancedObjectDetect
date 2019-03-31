"""
This module uses AI image recongition tools to determine whether or not specific
objects are present in a Blue Iris CCTV snapshot image, and if so, send a notification.
If objects are found, it further tries to determine whether they are stationary or have moved
since any previous notifications, to reduce false alerts due to e.g. parked cars.

At present, it can use one of:
    i.      Google's TensorFlow prebuilt model (from the TensorFlow zoo), running locally
    ii.     Amazon's AWS Rekognition cloud based solution
    iii.    SightHound's cloud based solution

Image recogition is triggered by receiving a notification via MQTT specifying the short name for
the camera in BlueIris. The module will then download a snapshot image for the specified camera,
and push this through the selected object recognition framework. Blue boxes are drawn around
objects found on the image, that meet the object search criteria of the camera. Yellow boxes are
drawn around objects that have been identified on a previous snapshot and are still in approximately
the same location. The processed image is saved in the designated folder for later review if required.

Once objects are detected, notifications from the module are sent back via:
    a.  MQTT (this can be picked up by BlueIris for further action)
    b.  Telegram
    c.  PushBullet

Note that the code has only been tested on Python 2.7. Additional libraries required include:

    Core libraries:
        -   cv2:            pip install python-opencv
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

from __future__ import print_function
import base64
import json
import ssl
import urllib
import time
import os
import signal
import sys
import httplib
from collections import namedtuple
import paho.mqtt.client as mqtt
import cv2
import numpy as np
import config


DEBUG = config.DEBUG

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


# Model names
TF = "TF"
SH = "SH"
AWS = "AWS"

Rectangle = namedtuple('Rectangle', 'xmin ymin xmax ymax')


class IdentifiedObject(object):
    ''' Object identified in an image'''

    def __init__(self, label=None, confidence=None, bounding_box=None, is_stationary=False):
        self.label = label
        self.confidence = confidence
        self.box = bounding_box
        self.is_stationary = is_stationary

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
        # download the image, convert it to a NumPy array, and then readit into OpenCV format
        if DEBUG:
            print("[DEBUG] Getting image from BlueIris url: %s" % url)

        resp = urllib.urlopen(url)
        image = np.asarray(bytearray(resp.read()), dtype="uint8")
        image = cv2.imdecode(image, cv2.IMREAD_UNCHANGED)
        self.timestamp = time.time()
        self.trigger_image = image
        self.processed_image = image    # Start off by having processed image same as initial image

        self._init_new_image()
        if DEBUG:
            print("[DEBUG] [ImageFrame.get_image_from_camera] Image width: {}, height: {}".format(
                self.width, self.height))
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
        print("[DEBUG][ImageFrame.load_image_from_file] Image width: {}, height: {}".format(
            self.width, self.height))
        # return the image
        return self.trigger_image

    def get_alert_objects_list(self, separator=", "):
        items = set([item.label for item in self.alert_objects])
        alert_list = separator.join(items)
        return alert_list

    def save_trigger_image(self, filename=None):
        if filename:
            self.trigger_file = filename

        if not self.trigger_file:
            self.trigger_file = os.path.join(
                config.IMAGE_SAVE_PATH, "%s_%ss.jpg" % (
                    time.strftime("%Y-%m-%d_%H.%M.%S", time.localtime(self.timestamp)),
                    self.camera_name))
        ret = cv2.imwrite(self.trigger_file, self.trigger_image)
        return ret

    def save_processed_image(self, filename=None):
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
    ''' Object holding details of Blue Iris camera    '''

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

    # --------------------------------------------------
    def download_trigger_image(self):
        if self.url is None:
            self.url = self.get_url()

        if self.url is not None and self.url:
            self.alert = ImageFrame(self.name)
            self.alert.get_image_from_camera(self.url)

            return self.alert.trigger_image
        else:
            print("[INFO ][download_trigger_image] Camera image download url not defined")
            return None

    # --------------------------------------------------
    def is_stationary(self, idx, label, pos):
        is_stationary = False
        idx += 1 # Aesthetics, as zero based
        if self.previous_objects:
            # Check if the objects are in roughly the same place
            for obj in self.previous_objects:
                prev_pos = obj.box
                overlap_percentage = rectangle_overlap_percentage(pos, prev_pos)
                if overlap_percentage > STATIONARY_OBJECT_OVERLAP_THRESHOLD:
                    # Object matched with previous location data, and is in approx same location
                    if DEBUG:
                        print("[DEBUG] [is_stationary] {} {}: Appears stationary ({:.2f}%) against " \
                            "prev object at ({}, {})/({}, {})".format(
                                obj.label, idx, overlap_percentage, prev_pos.xmin, prev_pos.ymin,
                                prev_pos.xmax, prev_pos.ymax))
                    is_stationary = True
                    break
                elif DEBUG:
                    print("[DEBUG] [is_stationary] {} {}: Ignoring {:.2f}% overlap against " \
                        "prev object at ({}, {})/({}, {})]".format(
                            obj.label, idx, overlap_percentage, prev_pos.xmin, prev_pos.ymin,
                            prev_pos.xmax, prev_pos.ymax))
        else:
            if DEBUG:
                print("[DEBUG] [is_stationary] Object {} {}: Ignoring stationary check as no "\
                    "previous object positions available".format(label, idx))

        return is_stationary

class AIObjectDetector(object):
    ''' Main AI detector object for the different AI frameworks '''

    def __init__(self, model=TF, image=None):
        self.model = model
        self.image = image
        self.cameras = {}
        self.active_cam = None

        if self.model == TF:
            print("Loading model from folder '%s' (may take some time)..." % self.model)
            # model_path = './faster_rcnn_inception_v2_coco_2018_01_28/frozen_inference_graph.pb'
            # model_path = './ssd_mobilenet_v1_coco_2017_11_17/frozen_inference_graph.pb'
            model_path = os.path.join(config.TENSORFLOW_MODEL_DIRECTORY, config.FROZEN_INFERENCE_GRAPH)
            self.tf_api = self.TFDetectorAPI(path_to_ckpt=model_path)
        elif self.model == SH:
            self.SIGHTHOUND_HEADERS = {"Content-type": "application/json",
                                       "X-Access-Token": config.SIGHTHOUND_CLOUD_TOKEN}

        elif self.model == AWS:
            print("Initiating AWS Rekognition client...")
            self.aws_client = boto3.client('rekognition')



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
                (obj.box.xmin + 2, obj.box.ymin - (text_height / 2)),
                cv2.FONT_HERSHEY_SIMPLEX, TAG_FONT_SCALE, text_colour, 1)

    # --------------------------------------------------
    def do_detect(self, cam_name):
        if DEBUG:
            print("[DEBUG] 'do_detect' function called for camera '%s'" % cam_name)

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
            print("[DEBUG] Downloadinge image from BlueIris")
        download_start_time = time.time()

        self.active_cam.download_trigger_image()
        self.active_cam.alert.detection_model = self.model

        # img = cv2.imread('./input/%s.jpg' % count, cv2.IMREAD_UNCHANGED)
        # img = cv2.resize(img, (1280, 720)) #Blue Iris image at 0.8 quality should be 1280x720 already

        # if DEBUG:
        print("[INFO ] Image downloaded from BI ({:.2f} seconds) [{} x {}]".format(
            time.time() - download_start_time, self.active_cam.alert.width, self.active_cam.alert.height))

        if self.model == TF:
            if DEBUG:
                print("[DEBUG] Calling 'process_tensorflow'")
            self.process_tensorflow()

        elif self.model == SH:
            if DEBUG:
                print("[DEBUG] Calling 'process_sighthound'")
            self.process_sighthound()

        elif self.model == AWS:
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
            # have bounding boxes and are moving/stationary
            for idx, obj in enumerate(self.active_cam.alert.detected_objects):
                obj.is_stationary = self.active_cam.is_stationary(idx, obj.label, obj.box)
                if not obj.is_stationary:
                    self.active_cam.alert.alert_objects.append(obj)

                # Draw box - this is after checking for stationary, so we get the right colours
                self.draw_bounding_box(obj)

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

        # with open(trigger_filename, 'rb') as image:
        #     print("[DEBUG] Posting image to AWS Rekognition API")
        #     response = client.detect_labels(Image={'Bytes': image.read()})

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
                    print("[DEBUG] -> {:<20}: {:.2f}% [ignored]".format(label['Name'], label['Confidence']))

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
        sighthound_connection = httplib.HTTPSConnection("dev.sighthoundapi.com",
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
            print("[DEBUG] TensorFlow: Classifying frame contents from camera '{}' "\
                "(may take some time depending on CPU/GPU)".format(self.active_cam.name))

        boxes, scores, classes, num = self.tf_api.process_frame(self.active_cam.alert.trigger_image)
        self.active_cam.alert.detected_objects = []
        if DEBUG:
            print("[DEBUG] Reviewing classfications for detected objects ({})".format(num))
        for i in range(len(boxes)):
            if len(self.TFDetectorAPI.COCO_LABELS) > classes[i]:
                # print("[DEBUG] classes[{}] value is {} ({})".format(
                #   i, classes[i], self.TFDetectorAPI.COCO_LABELS[classes[i]]))
                label = self.TFDetectorAPI.COCO_LABELS[classes[i]]
                # Check if the classfications found are in our wanted list
                confidence_level = scores[i] * 100
                if label in self.active_cam.search_objects and confidence_level >= config.CONFIDENCE_THRESHOLD:
                    obj = IdentifiedObject(label, confidence_level, boxes[i])
                    self.active_cam.alert.detected_objects.append(obj)
                elif DEBUG and confidence_level >= config.CONFIDENCE_THRESHOLD:
                    print("[DEBUG] -> {:<20} {:.2f} [ignored]"\
                        .format(self.TFDetectorAPI.COCO_LABELS[classes[i]], confidence_level))

        if DEBUG:
            print("[DEBUG] Completed TF processing. Returning to calling function")

    # TensorFlow class object --------------------------------
    class TFDetectorAPI(object):
        ''' TensorFlow API '''

        COCO_LABELS = ["Unknown", "Person", "Bicycle", "Car", "Motorcycle", "Airplane",
                       "Bus", "Train", "Lorry", "Boat", "Traffic light", "Fire hydrant",
                       "Stop sign", "Parking meter", "Bench", "Bird", "Cat", "Dog"]

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


#===============================================================================================================
def signal_handler(sig, frame):
    if mqtt_client:
        mqtt_client.loop_stop()
        mqtt_client.disconnect()
    print('Blue Iris Object Detection system terminated')
    sys.exit(0)

# Send notification --------------------------------
def send_notification(cam_name, image_file_name):
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
            print("[DEBUG] Sending telegram notification")

        telegram_notify.send_photo(
            chat_id=config.TELEGRAM_CHAT_ID,
            photo=open(image_file_name, 'rb'),
            caption="'" + cam_name  + "' motion detected")
        if DEBUG:
            print("[DEBUG] Telegram notification sent")
        notification_sent = True

    return notification_sent

# --------------------------------------------------
def rectangle_overlap_percentage(a, b):  # returns None if rectangles don't intersect
    dx = min(a.xmax, b.xmax) - max(a.xmin, b.xmin)
    dy = min(a.ymax, b.ymax) - max(a.ymin, b.ymin)
    # print("dx: {}, dy: {}".format(dx, dy))
    if (dx >= 0) and (dy >= 0):
        area_intersect = float(dx * dy)
        area_box1 = float((a.xmax - a.xmin) * (a.ymax - a.ymin))
        area_box2 = float((b.xmax - b.xmin) * (b.ymax - b.ymin))
        area_union = area_box1 + area_box2  - area_intersect
        return area_intersect/area_union * 100
    else:
        return 0.0


# MQTT Functions ---------------------------------
def mqtt_on_connect(client, userdata, flags, rc):
    if rc == 0:
        client.connected_flag = True #set flag
        print("MQTT connection established with broker")
    else:
        print("MQTT connection failed (code {})".format(rc))
        if DEBUG:
            print("[DEBUG] mqtt userdata: {}, flags: {}, client: {}".format(userdata, flags, client))

# --------------------------------------------------
def mqtt_on_disconnect(client, userdata, rc):
    client.loop_stop()
    if rc != 0:
        print("Unexpected disconnection.")
        if DEBUG:
            print("[DEBUG] mqtt rc: {}, userdata: {}, client: {}".format(rc, userdata, client))

# --------------------------------------------------
def mqtt_on_log(client, obj, level, string):
    if DEBUG:
        print("[DEBUG] MQTT log message received. Client: {}, obj: {}, level: {}".format(client, obj, level))
    print("[DEBUG] MQTT log msg: {}".format(string))


# --------------------------------------------------
def mqtt_on_message(client, userdata, msg):
    start_time = time.time()
    message = str(msg.payload)

    if config.BI_CLONED_CAMERA_COMMON_NAMING:
        cam_name = msg.payload.rsplit("_", 1)[0]
        if DEBUG and msg.payload.contains("_"):
            print("[DEBUG] Camera '{}' identified as clone of '{}'".format(msg.payload, cam_name))
    else:
        cam_name = msg.payload

    print(" ")
    print(time.strftime("%Y-%m-%d %H:%M:%S") + " '" + message + "' camera alert triggered")

    if DEBUG:
        print("[DEBUG] Calling image processing function...")
    detected = ai_detector.do_detect(cam_name)
    if detected:
        for obj in ai_detector.active_cam.alert.detected_objects:
            label = ("[stationary] " + obj.label) if obj.is_stationary else obj.label

            print("[INFO ] -> {:<20}: {:.2f}% (Bounding box: [{}, {}], [{}, {}])"\
                .format(label, obj.confidence, obj.box.xmin, obj.box.ymin, obj.box.xmax, obj.box.ymax))

        detected_items = ai_detector.active_cam.alert.get_alert_objects_list(", ")
        print("[INFO ] Motion/objects detected: " + detected_items + " ({0:.2f} seconds overall)"\
            .format(time.time() - start_time))

        if send_notification(ai_detector.active_cam.name, ai_detector.active_cam.alert.processed_file):
            print("------> Notification sent")
    else:
        print("[INFO ] Analysis complete. No motion detected ({0:.2f} seconds overall)"\
            .format(time.time() - start_time))





#===============================================================================================================
# Main --------------------------------------------
if __name__ == "__main__":
    signal.signal(signal.SIGINT, signal_handler)

    print("")
    print("======================================================")
    print("Person Detection system started")
    print("Modelling System: " + config.MODEL_SYSTEM)

    mqtt_client = None

    MODEL_TYPE = ""

    if config.MODEL_SYSTEM == AWS:
        print("Importing AWS Rekognition SDK")
        import boto3
        MODEL_TYPE = AWS

    elif config.MODEL_SYSTEM == SH:
        print("Initialising SightHound system")
        MODEL_TYPE = SH
    elif config.MODEL_SYSTEM == TF:
        full_path = os.path.join(config.TENSORFLOW_MODEL_DIRECTORY, config.FROZEN_INFERENCE_GRAPH)
        if not (os.path.isdir(config.TENSORFLOW_MODEL_DIRECTORY) and os.path.isfile(full_path)):
            print("[ERROR] Model inference graph file not found: '%s'" % full_path)
            sys.exit(0)
        print("Importing TensorFlow (may take a few seconds)")
        import tensorflow as tf
        MODEL_TYPE = TF
    else: # Default to tensorflow model
        print("Invalid modelling system. Please check the value in 'config.py' file")
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

    # Change to script folder
    abspath = os.path.abspath(__file__)
    dname = os.path.dirname(abspath)
    os.chdir(dname)

    # Instantiate main detector class (global)
    ai_detector = AIObjectDetector(MODEL_TYPE)

    # Set up MQTT subscriber - TODO!!! Error trapping....
    mqtt_client = mqtt.Client()
    mqtt_client.username_pw_set(config.MQTT_USER, config.MQTT_PW)
    mqtt_client.on_connect = mqtt_on_connect
    mqtt_client.on_message = mqtt_on_message
    # mqtt_client.on_log = mqtt_on_log
    mqtt_client.on_disconnect = mqtt_on_disconnect

    print("Model initalised. Connecting to mqtt server %s" % config.MQTT_SERVER)
    mqtt_client.connect(config.MQTT_SERVER, port=1883, keepalive=0, bind_address="")

    print("Subscribing to mqtt topic '%s'" % config.MQTT_LISTEN_TOPIC)
    mqtt_client.subscribe(config.MQTT_LISTEN_TOPIC)

    mqtt_client.loop_forever()
    # client.loop_start() #start loop to process received messages
