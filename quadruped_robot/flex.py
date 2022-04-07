import numpy as np
import RobotDART as rd
import dartpy  # OSX breaks if this is imported before RobotDART


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

print('Degrees of Freedom: {}'.format(robot.num_dofs()))
print('Joint Degrees of Freedom: {}'.format(robot.num_dofs()-6))
print('Degrees of Freedom per Joint: {}'.format((robot.num_dofs()-6)/4))
print('Available Joints: ' + ', '.join(robot.joint_names()[1:]))

# graphics.record_video("my_quadruped_robot.mp4")


counter = 0

while counter < 5000:
    simu.step_world()

    if counter == 3000:
        robot.set_positions([0.08, 0.08], ['FL_joint3', 'FR_joint3'])
    if counter == 2000:
        robot.set_positions([np.pi/4, np.pi/4, np.pi/4, np.pi/4], ['BL_joint1', 'BR_joint1', 'FL_joint1', 'FR_joint1'])
        robot.set_positions([-np.pi/2, -np.pi/2, -np.pi/2, -np.pi/2], ['BL_joint2', 'BR_joint2', 'FL_joint2', 'FR_joint2'])

    counter += 1
