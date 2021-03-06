# Description
This system allows you to make your robot partner learn your daily state (behavior).
The robot partner can be imagined as a virtual robot on your PC. (If you are the developer of the HSR robot, you can use it.)

The robot can observe the position of your face and the positions of surrounding objects and build a state recognition model based on the meaning and spatial location of those objects.

A demonstration of the system in use is shown below.

https://youtu.be/WI8vl_UJwCc

Here, we first have the robot record two states, "reading" and "working," to build a recognition model.
Then, we have the robot record a new state, "studying," and reconstruct the recognition model.

Recording of the states continues until 1000 pieces of data have been recorded, but the data recording can be terminated by the voice command "End". In the video, data recording is terminated by voice command after recording about 100 data.

As you can see from the video, all steps can be performed by voice commands as if interacting with a robot.


# Dependences

This project works on ROS system and depends on these 2 repo.

- https://github.com/leggedrobotics/darknet_ros

- https://github.com/ros-drivers/openni2_camera


Before clone this repo, Follow the installation instructions for each repository.

## Importnt Python library
Install the following libraries according to your PC environment.
When executing the code, you may find other libraries that need to be installed, in which case, please install them accordingly.
- pytorch geometric (https://pytorch-geometric.readthedocs.io/en/latest/notes/installation.html)
- pytorch
- fasttext (https://fasttext.cc/docs/en/support.html)
- pyaudio


# Install
```
$ cd catkin_ws/src
$ git clone git@github.com:Nao-Y1996/master_project.git
$ cd ..
$ catkin_make
$ source devel/setup.bash
```
before runing the code, you need to download fasttext model. 
(https://fasttext.cc/docs/en/crawl-vectors.html)

```
$ roscd master_project
$ cd script/w2v_model
$ python3 load_model.py
```

In this repo, speech2text client library is used.
So, follow the installation instructions (https://cloud.google.com/speech-to-text/docs/libraries#client-libraries-install-python)
and download a JSON key file.

Then, put the key file under  `master_project/script`

Then, you need to modify following code. 
- `master_project/script/speech2text.py`

At the line 12, change "/serene-star-322806-748861ca74d7.json" to JSON key file that you just downloaded
```
SERVICE_ACCOUNT_KEY = os.path.dirname(os.path.abspath(__file__)) +"/yourJSONkey.json"
```

# How to use
## Terminal
Connect the xtion camera to the PC and
(If using HSR robot, please start up the robot)
Launch the 7 terminals.

- Terminal 1

Object recognition node by YOLO, etc. is executed.
```
roslaunch master_project xtion.launch  # when xtion camera
roslaunch master_project hsr.launch  # when hsr robot
```

- Terminal 2

This is a node for pseudo interaction with the robot. Please enter your username when the node is launched.
The demo video uses HSR, which allows the robot to speak, making the interaction more realistic. when using the xtion camera, instead of the robot speaking, the terminal displays the robot's speech.
```
rosrun master_project interaction_server.py 
```

- Terminal 3

The speech recognition node that send your voice commands to the interaction_server node is executed.
```
rosrun master_project speech2text.py
```

- Terminal 4

This is the node that stores and distributes the data acquired by the camera.
```
rosrun master_project data_publisher.py 
```

- Terminal 5

This is a node that receives data and performs state recognition.
```
rosrun master_project data_subscriber.py 
```

- Terminal 6

This is a node that receives the results of state recognition and displays the results.
```
rosrun master_project probability_subscriber.py 
```

- Terminal 7

This is the node where the recognition model is trained.
```
rosrun master_project classification_nnconv_auto.py
```
## Voice Command
- ????????????????????????
    - you can check the robot mode.
- ??????????????????
    - you can start teaching your state.
- ?????????????????????????????????
    - you can execute training of GNN.
- ?????????????????????
    - you can change robot mode to recognition mode.

For other voice command, check script/interaction_server.py


# GNN
## Characteristics of the Recognition Model
The recognition model is built by a graph neural network.
The input is a graph that maps human faces and surrounding objects to nodes.
Node features: Meaning of objects
(300 dimensional word distributed representation of object names (human face is a word distributed representation of ???face???))
Edge features: positional relationship of nodes (3-dimensional relative position vector)

The recognition model is strongly influenced by the combination of objects.
On the other hand, even if the combination is exactly the same, it can be trained if the positional relationships are different (more data required for training)

## Network Configuration
GNN is using Edge-Conditioned Convolution (https://arxiv.org/abs/1704.02901)

For details, see
script/classification_nnconv_auto.py

