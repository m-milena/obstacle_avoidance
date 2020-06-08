import math
from tf.transformations import euler_from_quaternion, quaternion_from_euler
 
global counter

def pierwsza(data_file_name):
    global counter
    with open(data_file_name, 'a+') as myfile:
       for i in range(0, 300, 5):
           for j in range(0, 300, 5):
               dx = i/100.0
               dy = j/100.0
               napis = '%d,%.3f,%.3f,%.3f,%.3f,%.3f\n'%(counter, dx, dy, 0.000, 0.250, 0.000)
               myfile.write(napis)
               counter = counter + 1

def druga(data_file_name):
    global counter
    with open(data_file_name, 'a+') as myfile:
       for i in range(5, 300, 5):
           for j in range(0, 300, 5):
               dx = -i/100.0
               dy = j/100.0
               napis = '%d,%.3f,%.3f,%.3f,%.3f,%.3f\n'%(counter, dx, dy, 0.000, 0.250, 0.000)
               myfile.write(napis)
               counter = counter + 1

def trzecia(data_file_name):
    global counter
    with open(data_file_name, 'a+') as myfile:
       for i in range(0, 300, 5):
           for j in range(0, 300, 5):
               dx = -i/100.0
               dy = -j/100.0
               napis = '%d,%.3f,%.3f,%.3f,%.3f,%.3f\n'%(counter, dx, dy, 0.000, 0.250, 0.000)
               myfile.write(napis)
               counter = counter + 1

def czwarta(data_file_name):
    global counter
    with open(data_file_name, 'a+') as myfile:
       for i in range(5, 300, 5):
           for j in range(5, 300, 5):
               dx = -i/100.0
               dy = -j/100.0
               napis = '%d,%.3f,%.3f,%.3f,%.3f,%.3f\n'%(counter, dx, dy, 0.000, 0.250, 0.000)
               myfile.write(napis)
               counter = counter + 1


def stop(data_file_name):
    global counter
    with open(data_file_name, 'a+') as myfile:
        for i in range (-18000, 18000, 5):
            angle = i/100.0
            (orient_x, orient_y, orient_z, orient_w) = quaternion_from_euler(0, 0, angle)
            napis = '%d,%.3f,%.3f,%.3f,%.3f,%.3f\n'%(counter, 0.000, 0.000, angle, 0.000, 0.000)
            myfile.write(napis)
            counter = counter + 1

def normalize_angle(angle):
    if angle > 180:
        angle = -(360 - angle)
    elif angle < -180:
        angle = 360 + angle
    return angle

def orientation(data_file_name):
    global counter
    # pierwsza cwiartka
    with open(data_file_name, 'a+') as myfile:
        for i in range(-30, 30, 1):
           for j in range(-30, 30, 1):
               for k in range (-180, 180, 5):
                   angle = k #kat o jaki jestesmy obroceni
                   dx = i/10.0
                   dy = j/10.0

                   angle = normalize_angle(angle)
                   if angle > 0:
                       orient = 0.200
                   elif angle < 0:
                       orient = -0.200
                   else:
                       orient = 0
                   
                   if orient:
                       napis = '%d,%.3f,%.3f,%.3f,%.3f,%.3f\n'%(counter, dx, dy, angle, 0.000, orient)
                       myfile.write(napis)
                       counter = counter + 1   
                    

def main():
    data_file_name = "dataset_generated.txt"
    global counter
    counter = 0
    pierwsza(data_file_name)
    druga(data_file_name)
    trzecia(data_file_name)
    czwarta(data_file_name)
    stop(data_file_name)
    orientation(data_file_name)
    

if __name__ == '__main__':
    main()
