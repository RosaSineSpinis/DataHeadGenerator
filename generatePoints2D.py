import math
import matplotlib
import numpy as np

matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import meshing_function

class HalfSphere:
    PI = 3.14159265359
    r = 5
    N = 1000
    points_list = []

    def __init__(self, radius, particle_number):
        self.r = radius
        self.N = particle_number
        self.points_list = []
        self.points_list_numpy = None

        print("radius ", radius)
        print("particle_number ", particle_number)
        print("r ", self.r)
        print("N ", self.N)

    def generatePoints(self):
        """function creates points which create halh of sphere
        only this part of sphere which 'z' coordinate is bigger than 0"""

        print("r ", self.r)
        print("N ", self.N)

        it = 0;

        a = 4*self.PI*self.r*2 / self.N
        print("a ",a)
        d =pow(a, 1/2)
        print("d ", d)
        M_gamma = round(self.PI / d)
        print("M ", M_gamma)
        d_gamma = self.PI / M_gamma
        print("gamma ", d_gamma)
        d_phi = a / d_gamma
        print("phi ", d_phi)

        for m in range(0, M_gamma-1):
            gamma = self.PI * (m + 0.5)/M_gamma
            M_phi = round(2*self.PI*math.sin(gamma)/d_phi)
            for n in range(0, M_phi-1):
                phi = 2*self.PI*n/M_phi
                x = self.r*math.sin(gamma)*math.cos(phi)
                y = self.r*math.sin(gamma)*math.sin(phi)
                z = math.cos(gamma)
                it += 1
                if z > 0:
                    self.points_list.append((x,y,z))

        self.points_list_numpy = np.array(self.points_list)
        return self.points_list


obj = HalfSphere(5, 4000)
obj.generatePoints()


obj2 = HalfSphere(8, 4000)
obj2.generatePoints()

# fig = plt.figure()
# ax = Axes3D(fig)

# fig = plt.figure()
# ax = fig.add_subplot(111, projection='3d')
#
# for x in obj.points_list:
#     ax.scatter(x[0], x[1], x[2], s=10, c='r', marker='o')
#     # ax.axes.scatter(x[0], x[1], x[2], s=10, c='r', marker='o')
#
#
# for y in obj2.points_list:
#     ax.scatter(y[0], y[1], y[2], s=10, c='b', marker='x')
#     # ax.axes.scatter(y[0], y[1], y[2], s=10, c='b', marker='x')
#
#
#
# plt.show()


# print("type np array ", type(obj2.points_list_numpy))
# print(obj2.points_list_numpy)

meshed_half_sphere = meshing_function.generate_mesh(obj2.points_list_numpy)

meshing_function.plot_mesh(obj2.points_list_numpy, meshed_half_sphere)

