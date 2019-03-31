# BlueIris Enhanced Object Detection using Deep Learning AI Tools 


BlueIris is a comprehensive CCTV management tool. However, whilst its object and motion detection is better than most in its category, it does suffer similar false alert issues arising from changing shadows, clouds, rain, trees etc that all such simple pixel change motion/object detection algorithms suffer from.

The python script here tries to improve the accuracy of BlueIris' object detection by running a snapshot image from BlueIris through one of the following machine learning/deep learning tools:

1. [Google TensorFlow](https://www.tensorflow.org/) framework running locally with pre-built models. The components of this solution are freely availble, and there is no other cost in running this. However, its speed of recognition will depend on (a) the model type/size being used (b) the CPU and/or GPU of the system running the script.

1. [Amazon's "Rekogntion"](https://aws.amazon.com/rekognition/) image analysis service, which is run on the Amazon AWS servers and thus in the cloud (image is uploaded to Rekogntion, which then sends back a JSON response with details of what it recognised). Currently Amazon gives 5000 free calls a month for the first 12 months of subscription, after which there is a charge.

1. [SightHound's cloud API](https://www.sighthound.com/). At present, SightHound only seems to do either people detection or vehicle detection via their API, but not both at the same time. As such, I no longer use this, but have left in the code in case anyone finds it useful. It has not been tested recently and so may or may not still be working.


#### TensorFlow solution
The TensorFlow solution uses pre-built models from the [Tensorflow detection model zoo](https://github.com/tensorflow/models/blob/master/research/object_detection/g3doc/detection_model_zoo.md). Various models are available in the "zoo" along with their speed/accuracy data. I have focused on using the following 2 models:
1. [faster_rcnn_inception_v2_coco](http://download.tensorflow.org/models/object_detection/faster_rcnn_inception_v2_coco_2018_01_28.tar.gz). This is a slightly slower model but more accurate. Using a Xeon 1260L processor running an Ubuntu VM (and no GPU), I typically get recogntion in about 2 to 2.5 seconds.

1. [ssd_mobilenet_v2_coco](http://download.tensorflow.org/models/object_detection/ssd_mobilenet_v2_coco_2018_03_29.tar.gz). This works well with low power CPUs, responding in under a second on my above VM. However, due to its reduced accuracy, for my purposes I do not use this model except during development/testing for increased speed.

With a good GPU, the TensorFlow approach could provide realtime detection from a streaming mjpg feed from BlueIris.

#### AWS Rekognition
[Amazon's "Rekogntion"](https://aws.amazon.com/rekognition/) is a cloud based solution, that provides good object detection, without needing high power CPUs/GPUs on the local PC. Typical response times using this method on my VM has been of the order of about 0.75 to 1.25 seconds. As such, this provides a faster response than the TensorFlow model with my set up.

If using this method for object detection, it will need to be configured such that it can be accessed from its command line interface (remembering to save the credentials and region details into the ~/.aws directory as detailed in the Amazon documentation)


### How well does it work?

For my set up, false alerts are now down to 1 or 2 a day, if that (and on many days, zero false alerts), compared to 20+ a day. It has only been tested over the last month or so, and so time will tell how effective it is with the changing seasons/lighting conditions etc.

### Miscellaneous
Python is not my main coding language and as such the code in the script could most likely benefit from tidying up. However,  this script was written to solve a problem that I was having, and is currently at a point where it is working for me. As such, going forward I am not intending to actively look further at this script, but welcome any pull requests that enhance the functionality, improve the coding etc. 

## Installation:

### Server Side
The script has been developed on Python 2.7, installed on an Ubuntu 16.04 server (running on an ESXi virtual machine). In addition to python 2.7, the following libraries are required:

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

In addition, for TensorFlow, an appropriate prebuilt model from the TensorFlow zoo is required as discussed previously. The whole folder containing this model needs to be placed in the root folder with this script (or its directory path needs to be updated in the `config.py` file discussed below)

A copy of a `systemd` service file is included, which demonstrates running the script as a service on a linux distribution that uses `systemd`.

Finally, copy the `config.sample` file to `config.py` and update/save the required parameters (in particular the BlueIris details).

### Interfacing with Blue Iris
Intefacing with BlueIris is via an mqtt broker, such as [mosquitto](https://mosquitto.org/download/). 

1. First, BlueIris first needs to have its mqtt settings configured (under Setttings -> Digital IO and IoT -> MQTT). 

2. Each camera that is to have its alerts processed through this script needs to individually have its alert settings configured such that it publishes the camera name to the defined mqtt topic (default as in the sample config is `BlueIris/alert`). This is done via Camera Properties -> Alerts tab and then click on _Post to a Web Adress or MQTT Server_ button. On the pop-up, under the _When Triggered_ section, select `MQTT topic` from the dropdown and enter the topic name that the script is monitoring, e.g. `BlueIris/alert`. In the box below this, for the _Post/payload_, enter `&CAM`. 

3. The above needs to be repeated for each camera.

Once the AI script has done its processing, if you want BlueIris to respond (e.g. to trigger a recording, or send a rich gif notification), you will need to:

1.  create a copy of the camera so that there is a new camera that does any of the post-processing you want. The main steps to do this are (a) clone the camera (b) DISABLE everything in the trigger section of the *cloned* camera (we will be using the mqtt admin topic to trigger) and (c) set up notifications or recording as you would normally (but obviously _not_ sending anything to the AI script).  

5. You will also need to enable BlueIris notifications in the `config.py` file of the script, so that the script posts to the  BlueIris admin topic, in turn triggerring the cloned camera in BlueIris.


#### Credits
Initial inspiration, particularly for the TensorFlow side of things, came from  https://medium.com/@madhawavidanapathirana/real-time-human-detection-in-computer-vision-part-2-c7eda27115c6.


