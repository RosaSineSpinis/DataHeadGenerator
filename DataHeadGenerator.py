import os
import json
import matplotlib
matplotlib.use('TkAgg')
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider, Button, RadioButtons
import cmath
import math

print("We are starting the game")

'''
loaded_file = open(path, "r")
f = loaded_file.readlines() #here we create the list
for x in f: #and here we go through all the lines of the list
    print(x)

loaded_file.close()
'''

def loadFileToList(string):
    list_from_file = []
    loaded_file = open(path, "r")
    for x in loaded_file:
        #print("line = " + x)
        list_from_file.append(x)
    loaded_file.close()
    return list_from_file


def generatePoints(beginning_pos_x, ending_pos_x, beginning_pos_y, ending_pos_y, beginning_pos_z, ending_pos_z,
                   increment_x, increment_y, increment_z, small_radius, big_radius):
    points_list = []
    x = beginning_pos_x
    y = beginning_pos_y
    z = beginning_pos_z
    while x <= ending_pos_x:
        while y <= ending_pos_y:
            while z <= ending_pos_z:
                root_x_y = math.sqrt(x**2 + y**2)
                root_x_z = math.sqrt(x**2 + z**2)
                root_y_z = math.sqrt(y**2 + z**2)
                if small_radius < root_x_y < big_radius:
                    points_list.append(tuple([x, y, z]))
                '''
                if small_radius < root_x_z < big_radius:
                    points_list.append((x, y, z))
                if small_radius < root_y_z < big_radius:
                    points_list.append((x, y, z))
                '''
                z = z + increment_z
            y = y + + increment_y
            z = beginning_pos_z
        y = beginning_pos_y
        x = x + increment_x
    for i in points_list:
        print(i)
    return points_list


def saveListToFile(path, input_list):
    save_file = open(path, "w+")
    for x in input_list:
        save_file.write(x)
    save_file.close()
    return

'''d
def update(val):
    amp = x_min_values.val
    fig.canvas.draw_idle()
'''
'''
    for x in range(0, 10):
        for y in range(0, 10):
            points_list.append([x, y])
    for i in points_list:
        print(i)
'''



generated_points_list = []
generated_points_list = generatePoints(-6, 6, -6, 6, -6, 6, 2, 2, 2, 5, 7)


path = os.path.join('.', 'input_data.txt')
point_list = []
point_list = loadFileToList(path)
output_loaded_data_name = os.path.join('.', 'output_loaded_data_name.txt')
output_generated_data_name = os.path.join('.', 'output_generated_data_name.txt')

saveListToFile(output_loaded_data_name, point_list)#just to check, wheter file is loaded correctly

save_file = open(output_generated_data_name, "w+")
json.dump(generated_points_list, save_file)
save_file.close()



#modify this function and check with 2d plot

# fig, ax = plt.subplots()
# fig.subplots_adjust(bottom=0.3)
#ax = plt.subplot(projection='3d')

# fig, ax = plt.subplots(subplot_kw={'projection': '3d'})
# fig.subplots_adjust(bottom=0.3)
#
# x_val = [t[0] for t in generated_points_list]
# y_val = [t[0] for t in generated_points_list]
# z_val = [t[0] for t in generated_points_list]
#
# print("x_val ", x_val)
# print("y_val ", y_val)
# print("z_val ", z_val)
#
# ax.scatter(x_val, y_val, z_val, s=10, c='r', marker='o')

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

for x in generated_points_list:
    ax.scatter(x[0], x[1], x[2], s=10, c='r', marker='o')
    # ax.axes.scatter(x[0], x[1], x[2], s=10, c='r', marker='o')


print("going to loop")
# for x in generated_points_list:
#     # ax.axes.scatter(x[0], x[1], s = 10, c='r', marker='o')
#     ax.scatter(x[0], x[1], x[2], s = 10, c='r', marker='o')
#     print("loop")
'''
def plot_graph(list_of_points):
    plt.subplots()
    plt.subplots_adjust(bottom=0.3)
    ax = plt.subplot(projection='3d')
    for x in list_of_points:
        ax.scatter(x[0], x[1], x[2], c='r', marker='o')
    return ax


canvas = plot_graph(generated_points_list)


def plot_graph2(list_of_points, ax):

    for x in list_of_points:
        ax.set_xdata(x[0])
        ax.set_ydata(x[1])
        ax.set_zdata(x[2])
    plt.draw()
    return ax
'''

#ax = fig.add_subplot(1, 1, 1, projection='3d')
#plt.axis
#ax = Axes3D(fig)
#ax = fig.add_subplot(1, 1, 1, projection='3d')


axcolor = 'lightgoldenrodyellow'
slider_x_min = plt.axes([0.1, 0.15, 0.3, 0.02], facecolor=axcolor)
x_min = Slider(slider_x_min, 'x_min', valmin=-100, valmax=100, valstep=1, valinit=-50)
#x_min_values.on_changed(update)

axcolor = 'lightgoldenrodyellow'
slider_x_max = plt.axes([0.6, 0.15, 0.3, 0.02], facecolor=axcolor)
x_max = Slider(slider_x_max, 'x_max', valmin=-100, valmax=100, valstep=1, valinit=50)

axcolor = 'lightgoldenrodyellow'
slider_y_min = plt.axes([0.1, 0.1, 0.3, 0.02], facecolor=axcolor)
y_min = Slider(slider_y_min, 'y_min', valmin=-100, valmax=100, valstep=1, valinit=-50)

axcolor = 'lightgoldenrodyellow'
slider_y_max = plt.axes([0.6, 0.1, 0.3, 0.02], facecolor=axcolor)
y_max = Slider(slider_y_max, 'y_max', valmin=-100, valmax=100, valstep=1, valinit=50)

axcolor = 'lightgoldenrodyellow'
slider_z_min = plt.axes([0.1, 0.05, 0.3, 0.02], facecolor=axcolor)
z_min = Slider(slider_z_min, 'z_min', valmin=-100, valmax=100, valstep=1, valinit=-50)

axcolor = 'lightgoldenrodyellow'
slider_z_max = plt.axes([0.6, 0.05, 0.3, 0.02], facecolor=axcolor)
z_max = Slider(slider_z_max, 'z_max', valmin=-100, valmax=100, valstep=1, valinit=50)

#def val_update(val):
    #generated_points_list = generatePoints(x_min.val, x_max.val, y_min.val, y_max.val, z_min.val, z_max.val, 1, 1, 1, 5, 7)
   #plot_graph(generated_points_list)

# yval = x_min.val
    #plt.draw()


#x_min.on_changed(val_update)
#x_max.on_changed(val_update)

recompute_button_box = plt.axes([0.1, 0.9, 0.25, 0.1])
recompute_button = Button(recompute_button_box, 'Recompute')

print('we are here')


def recompute(val):
    print('x_min ' + str(x_min.val) + ' x_max= ' + str(x_max.val) + ' y_min= ' + str(y_min.val) + ' y_max= ' + str(y_max.val) +
          ' z_min= ' + str(z_min.val) + ' z_max= ' + str(z_max.val))

    global generated_points_list
    generated_points_list = generatePoints(x_min.val, x_max.val, y_min.val, y_max.val, z_min.val, z_max.val, 2, 2, 2, 5, 7)
    for i in generated_points_list:
        print(i)
    # ax.axes.clear()
    ax.clear()
    for x in generated_points_list:
        print(str(x[0]) + ' ' + str(x[1]) + ' ' + str(x[2]))
        # ax.axes.scatter(x[0], x[1], s = 10, c='r', marker='o')
        ax.scatter(x[0], x[1], s = 10, c='r', marker='o')

    plt.show()


recompute_button.on_clicked(recompute)

print('OUTSIDE \nx_min ' + str(x_min.val) + ' x_max= ' + str(x_max.val) + '\n y_min= ' + str(y_min.val) + ' y_max= ' +
      str(y_max.val) + '\n z_min= ' + str(z_min.val) + ' z_max= ' + str(z_max.val))

plt.show()
