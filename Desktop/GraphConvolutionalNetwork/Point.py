import numpy as np


def latlong_to_xyz(lat,long,r):
    x = r * np.cos(lat) * np.cos(long)
    y = r * np.cos(lat) * np.sin(long)
    z = r * np.sin(lat)
    return x, y, z

class Point():
    def __init__(self, lat, long):
        self.lat = lat
        self.long = long
        self.x, self.y, self.z = latlong_to_xyz(lat, long, 1)