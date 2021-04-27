# vehicle_recognition

Vehicle Type Recognition and Vehicle Color Recognition

## Vehicle Type Recognition

Help was taken from https://github.com/hoanhle/Vehicle-Type-Detection for this part. All due credits to the original owner. The vehicle image is classified into one of the following:
1. Ambulance
2. Barge
3. Bicycle
4. Boat
5. Bus
6. Car
7. Cart
8. Caterpillar
9. Helicopter
10. Limousine
11. Motorcycle
12. Segway
13. Snowmobile
14. Tank
15. Taxi
16. Truck
17. Van

Make sure all the requirements are installed as specified in [requirements.txt](vehicle_type_recognition/requirements.txt) file. Then run [api_server.py](vehicle_type_recognition/api_server.py) file to run the server. You can make a `POST` request at the specified server as shown in the terminal when you run [api_server.py](vehicle_type_recognition/api_server.py) file. Attach the image whose vehicle type needs to be recognized in the `POST` request with the key `image`. The response is a json text with key `result` containing the prediction.

The trained models can be found in `vehicle_type_recognition` folder at https://drive.google.com/drive/folders/1iBAn9IwWXY8Ur4JA89ZkIOP4MSjtDea0?usp=sharing and these models (namely `densenet.h5`, `inception_v3.h5` and `mobilenet_v2.h5`) should be downloaded and put inside `vehicle_type_recognition/models` folder. We can change the path of the folder which contains the models by changing the `app.config["MODELS_PATH"]` configuration of the flask app (currently, this configuration is set inside [api_server.py](vehicle_type_recognition/api_server.py) file).

## Vehicle Color Recognition

Visit the [VehicleColorRecognition](https://github.com/i-am-g2/VehicleColorRecognition) repository for details of Vehicle Color Recognition part.
