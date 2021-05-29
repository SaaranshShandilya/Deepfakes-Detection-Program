

## Getting Started
In order to run the program, you need to install an IDE like Visual Studio Code or Anaconda application that provides a base for the code.
You also need to install python version 3.8.3 on your machine and hence set environment variables if needed for the same. 

## Installation
If you are using Anaconda then open Anaconda Command prompt else open windows command prompt.Using the pip install (package name) command you need to install the following packages:
1. numpy
2. tensorflow
3. opencv--python 
4. os
5. matplotlib.pyplot

## Functionality
In order to determine whether a given video is deepfake or not, the code first splits the videos into frames. Then it scans for a face on each frame.
It crops the the face, stores it in another folder, which is then accessed by Meso4 software that checks for the probability of the given face being a deepfake or not.
The closer the probability is to 1, the more chances are there that the given video is a deepfake and vice-versa.
The program also checks whether the given prediction is reliable or not by checking if the probable value and absolute value are same when rounded off.

## Tech Stack


## Deliverables
- Code files
- Documentation 

## Demonstration
<img src="Screencast/Preview.gif" alt="Sample">
For further references and complete tutorial follow the given drive link to watch the video.
https://drive.google.com/file/d/1qbC69TAIASpRbnWbhjRVKUtL1PwnMBX4/view?usp=sharing

### Issues
Have a bug or any other problem? Please first read the issues and search for existing and closed issues. If your problem or idea is not addressed yet, please open a new issue in issue section. 

Also, try to refer content from screen record or internet incase if this is a trivial issue. 


## Creators
Saaransh Shandilya




