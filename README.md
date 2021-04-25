# vehicle_recognition

Vehicle Type Recognition and Vehicle Color Recognition

## Vehicle Type Recognition

Help was taken from https://github.com/hoanhle/Vehicle-Type-Detection for this part. The vehicle image is classified into one of the following:
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

Make sure all the requirements are installed as specified in [requirements.txt](vehicle_type_recognition/requirements.txt) file. Then run [api_server.py](vehicle_type_recognition/api_server.py) file to run the server. You can make a **POST** request at the specified server as shown in the terminal when you run [api_server.py](vehicle_type_recognition/api_server.py) file. Attach the image whose vehicle type needs to be recognized in the **POST** request with the key **image**. The response is a json text with key **result** containing the prediction.

## Vehicle Color Recognition
