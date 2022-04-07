import numpy as np
import RobotDART as rd
import dartpy  # OSX breaks if this is imported before RobotDART
from numpy import linalg as LA

robot = rd.Robot("robot.urdf")
robot.set_position_enforced(True)
robot.set_actuator_types("servo")

timestep = 0.001
simu = rd.RobotDARTSimu(timestep)
simu.set_collision_detector("fcl")

gconfig = rd.gui.GraphicsConfiguration(1024, 768)
graphics = rd.gui.Graphics(gconfig)
simu.set_graphics(graphics)
graphics.look_at([2, -3, 2], [0, 0, 0])

simu.add_robot(robot)
simu.add_checkerboard_floor()

robot.set_base_pose([0, 0, 0, 0, 0, 1])


# wait untill the robot reaches the floor
counter = 0
while counter < 1000:
    simu.step_world()
    counter += 1


def calculate_fk(foot):
    # My Forward Kinematics
    joints = robot.positions([foot+'_joint1', foot+'_joint2', foot+'_joint3'])
    L0 = 0.15
    L1 = 0.2
    # the third part of the foot consists of a box and a shpere
    # box's height is 0.1
    # sphere's diameter is 0.8 and it's center is 0.02 below the end of the box
    L2 = 0.1 + 0.02 + 0.08

    # position of the first join of the foot
    tf_01 = dartpy.math.Isometry3()
    x_translation = 0.225 if foot[1] == 'R' else -0.225
    y_translation = 0.325 if foot[0] == 'F' else -0.325
    tf_01.set_translation(robot.body_pose("body").translation() + [x_translation, y_translation, -0.1])

    tf_12 = dartpy.math.Isometry3()
    tf_12.set_rotation(dartpy.math.eulerZYXToMatrix([0., 0., joints[0]]))
    tf_12.set_translation([0, 0, -L0])

    tf_23 = dartpy.math.Isometry3()
    tf_23.set_rotation(dartpy.math.eulerZYXToMatrix([0., 0., joints[1]]))
    tf_23.set_translation([0, 0, -L1])

    tf_34 = dartpy.math.Isometry3()
    tf_34.set_translation([0, 0, -L2+joints[2]]) # this is a prismatic joint


    tf = tf_01.multiply(tf_12).multiply(tf_23).multiply(tf_34)
    # print("My Forward Kinematics")
    # print(tf)

    # Robot Dart Forward Kinematics
    # print("Robot Dart Forward Kinematics:")
    # print(robot.body_pose(foot+"_foot4"))


    print("Norm of Difference: {}".format(LA.norm((tf.matrix() - robot.body_pose(foot+"_foot4").matrix()))))


for x in ['FR', 'FL', 'BR', 'BL']:
    calculate_fk(x)


while True:
    simu.step_world()
