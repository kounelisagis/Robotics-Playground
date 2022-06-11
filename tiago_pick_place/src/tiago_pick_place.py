import RobotDART as rd
import dartpy  # OSX breaks if this is imported before RobotDART
from utils import create_grid, box_into_basket, damped_pseudoinverse
import numpy as np
import py_trees


class PITask:
    def __init__(self, target, dt, Kp, Ki):
        self._target = target
        self._dt = dt
        self._Kp = Kp
        self._Ki = Ki
        self._sum_error = 0
    
    def set_target(self, target):
        self._target = target
    
    def error(self, tf):
        rot_error = rd.math.logMap(self._target.rotation() @ tf.rotation().T)
        lin_error = self._target.translation() - tf.translation()
        return np.r_[rot_error, lin_error]
    
    def update(self, current):
        error_in_world_frame = self.error(current)

        self._sum_error = self._sum_error + error_in_world_frame * self._dt

        return self._Kp * error_in_world_frame + self._Ki * self._sum_error


class ReachArmTarget(py_trees.behaviour.Behaviour):
    def __init__(self, robot, goal, offset, dt, goal_bubble, finger_velocity_end, Kp, Ki, name="ReachArmTarget"):
        """
        offset: 3d vector
        """

        super(ReachArmTarget, self).__init__(name)

        self.robot = robot
        self.goal = goal
        self.offset = offset
        self.goal_bubble = goal_bubble
        # end-effector name
        self.eef_link_name = "gripper_link"

        self.dt = dt
        self.arm_dofs = ["arm_1_joint", "arm_2_joint", "arm_3_joint", "arm_4_joint", "arm_5_joint", "arm_6_joint", "arm_7_joint"]
        self.finger_velocity_end = finger_velocity_end
        self.Kp = Kp # Kp could be an array of 6 values
        self.Ki = Ki # Ki could be an array of 6 values

        self.logger.debug("%s.__init__()" % (self.__class__.__name__))

    def setup(self):
        self.logger.debug("%s.setup()->does nothing" % (self.__class__.__name__))

    def initialise(self):
        self.logger.debug("%s.initialise()->init controller" % (self.__class__.__name__))

        self.tf_desired = dartpy.math.Isometry3()
        self.tf_desired.set_translation(self.goal.base_pose().translation() + self.offset)
        self.tf_desired.set_rotation(self.goal.base_pose().rotation())

        self.controller = PITask(self.tf_desired, self.dt, self.Kp, self.Ki)

        self.goal_bubble.set_base_pose(self.controller._target)

    def update(self):
        new_status = py_trees.common.Status.RUNNING
        # control the robot
        tf = self.robot.body_pose(self.eef_link_name)
        vel = self.controller.update(tf)
        jac = self.robot.jacobian(self.eef_link_name) # this is in world frame
        jac_pinv = damped_pseudoinverse(jac)
        cmd = jac_pinv @ vel
        cmd = self.robot.vec_dof(cmd, self.arm_dofs)
        self.robot.set_commands(cmd, self.arm_dofs)

        # if error too small, report success
        err = np.linalg.norm(self.controller.error(tf))
        if err < 1e-2:
            new_status = py_trees.common.Status.SUCCESS

        if new_status == py_trees.common.Status.SUCCESS:
            self.feedback_message = "Reached target"
            self.logger.debug("%s.update()[%s->%s][%s]" % (self.__class__.__name__, self.status, new_status, self.feedback_message))
        else:
            self.feedback_message = "Error: {0}".format(err)
            self.logger.debug("%s.update()[%s][%s]" % (self.__class__.__name__, self.status, self.feedback_message))
        return new_status

    def terminate(self, new_status):
        if new_status == py_trees.common.Status.SUCCESS:
            self.robot.set_commands([self.finger_velocity_end], ['gripper_finger_joint'])

        self.logger.debug("%s.terminate()[%s->%s]" % (self.__class__.__name__, self.status, new_status))


class ReachWheelTarget(py_trees.behaviour.Behaviour):
    def __init__(self, robot, goal, offset, dt, goal_bubble, name="ReachWheelTarget"):
        """
        offset: number
        """
        super(ReachWheelTarget, self).__init__(name)
        # robot
        self.robot = robot
        self.goal = goal
        self.offset = offset
        self.goal_bubble = goal_bubble
        # end-effector name
        self.base_link_name = "base_link"

        self.dt = dt
        self.wheel_dofs = ['rootJoint_rot_z', 'rootJoint_pos_x', 'rootJoint_pos_y']

        self.logger.debug("%s.__init__()" % (self.__class__.__name__))

    def setup(self):
        self.logger.debug("%s.setup()->does nothing" % (self.__class__.__name__))

    def initialise(self):
        
        self.logger.debug("%s.initialise()->init controller" % (self.__class__.__name__))

        vec_desired = self.goal.base_pose_vec()
        z = np.arctan2(vec_desired[4], vec_desired[3])
        distance = np.linalg.norm([vec_desired[3], vec_desired[4]])
        distance += self.offset
        vec_desired[3] = distance * np.cos(z)
        vec_desired[4] = distance * np.sin(z)
        self.tf_desired = dartpy.math.Isometry3()
        self.tf_desired.set_translation(vec_desired[3:])
        self.tf_desired.set_rotation(dartpy.math.eulerZYXToMatrix([z,0,0]))

        self.Kp = 0.5 # Kp could be an array of 6 values
        self.Ki = 0.001 # Ki could be an array of 6 values
        self.controller = PITask(self.tf_desired, self.dt, self.Kp, self.Ki)

        self.goal_bubble.set_base_pose(self.controller._target)

    def update(self):
        new_status = py_trees.common.Status.RUNNING
        # control the robot
        tf = self.robot.body_pose(self.base_link_name)
        vel = self.controller.update(tf)
        jac = self.robot.jacobian(self.base_link_name) # this is in world frame
        jac_pinv = damped_pseudoinverse(jac)
        cmd = jac_pinv @ vel
        cmd = self.robot.vec_dof(cmd, self.wheel_dofs)
        self.robot.set_commands(cmd, self.wheel_dofs)

        # if error too small, report success
        err = np.linalg.norm(self.controller.error(tf))
        if err < 1e-1:
            new_status = py_trees.common.Status.SUCCESS

        if new_status == py_trees.common.Status.SUCCESS:
            self.feedback_message = "Reached target"
            self.logger.debug("%s.update()[%s->%s][%s]" % (self.__class__.__name__, self.status, new_status, self.feedback_message))
        else:
            self.feedback_message = "Error: {0}".format(err)
            self.logger.debug("%s.update()[%s][%s]" % (self.__class__.__name__, self.status, self.feedback_message))
        return new_status

    def terminate(self, new_status):
        if new_status == py_trees.common.Status.SUCCESS:
            self.robot.set_commands([0.]*len(self.wheel_dofs), self.wheel_dofs)

        self.logger.debug("%s.terminate()[%s->%s]" % (self.__class__.__name__, self.status, new_status))


class WaitState(py_trees.behaviour.Behaviour):
    def __init__(self, box, basket, name="WaitState"):
        super(WaitState, self).__init__(name)
        self.box = box
        self.basket = basket

        self.logger.debug("%s.__init__()" % (self.__class__.__name__))

    def setup(self):
        self.logger.debug("%s.setup()->does nothing" % (self.__class__.__name__))

    def initialise(self):
        self.logger.debug("%s.initialise()->init controller" % (self.__class__.__name__))

    def update(self):
        new_status = py_trees.common.Status.RUNNING

        box_translation = self.box.base_pose().translation()
        basket_translation = self.basket.base_pose().translation()
        if box_into_basket(box_translation, basket_translation, 0.):
            new_status = py_trees.common.Status.SUCCESS

            clone_move_box(box_positions, box)
            # finished += 1

        if new_status == py_trees.common.Status.SUCCESS:
            self.feedback_message = "Reached target"
            self.logger.debug("%s.update()[%s->%s][%s]" % (self.__class__.__name__, self.status, new_status, self.feedback_message))
        else:
            self.feedback_message = "Waiting"
            self.logger.debug("%s.update()[%s][%s]" % (self.__class__.__name__, self.status, self.feedback_message))
        return new_status

    def terminate(self, new_status):
        self.logger.debug("%s.terminate()[%s->%s]" % (self.__class__.__name__, self.status, new_status))



def clone_move_box(box_positions, box):
    simu.add_robot(box.clone())
    # Create box
    box_size = [0.04, 0.04, 0.04]
    # Random cube position
    box_pt = np.random.choice(len(box_positions))
    box_pose = [0., 0., 0., box_positions[box_pt][0], box_positions[box_pt][1], box_size[2] / 2.0]
    box.set_base_pose(box_pose)



dt = 0.001
simulation_time = 500.
total_steps = int(simulation_time / dt)

# Create robot
packages = [("tiago_description", "tiago/tiago_description")]
robot = rd.Tiago(int(1. / dt), "tiago/tiago_steel.urdf", packages)
arm_dofs = ["arm_1_joint", "arm_2_joint", "arm_3_joint", "arm_4_joint", "arm_5_joint", "arm_6_joint", "arm_7_joint", "gripper_finger_joint", "gripper_right_finger_joint"]
robot.set_positions(np.array([np.pi/2., np.pi/4., 0., np.pi/2., 0. , 0., np.pi/2., 0.03, 0.03]), arm_dofs)

# Control base - we make the base fully controllable
robot.set_actuator_type("servo", "rootJoint", False, True, False)

# Create position grid for the box/basket
basket_positions, box_positions = create_grid()

box_size = [0.04, 0.04, 0.04]
# Random cube position
box_pt = np.random.choice(len(box_positions))
box_pose = [0., 0., 0., box_positions[box_pt][0], box_positions[box_pt][1], box_size[2] / 2.0]
box = rd.Robot.create_box(box_size, box_pose, "free", 0.1, [0.9, 0.1, 0.1, 1.0], "box_" + str(0))

# Create basket
basket_packages = [("basket", "models/basket")]
basket = rd.Robot("models/basket/basket.urdf", basket_packages, "basket")
# Random basket position
basket_pt = np.random.choice(len(basket_positions))
basket_z_angle = 0.
basket_pose = [0., 0., basket_z_angle, basket_positions[basket_pt][0], basket_positions[basket_pt][1], 0.0008]
basket.set_positions(basket_pose)
basket.fix_to_world()

goal_bubble = rd.Robot.create_ellipsoid(dims=[0.15, 0.15, 0.15], color=[0., 1., 0., 0.5], ellipsoid_name="target")


# Behavior Tree
py_trees.logging.level = py_trees.logging.Level.DEBUG

# Create sequence node
sequence = py_trees.composites.Sequence(name="Sequence")

trA = ReachWheelTarget(robot, box, -0.5, dt, goal_bubble, "Reach Target A")
sequence.add_child(trA)

trB = ReachArmTarget(robot, box, [0., 0, 0.15], dt, goal_bubble, -0.4, 15, 0.2, "Reach Target B")
sequence.add_child(trB)

trC = ReachArmTarget(robot, box, [0., 0, 0.4], dt, goal_bubble, -0.4, 15, 0.5, "Reach Target C")
sequence.add_child(trC)

trD = ReachWheelTarget(robot, basket, -0.5, dt, goal_bubble, "Reach Target D")
sequence.add_child(trD)

trE = ReachArmTarget(robot, basket, [0., 0., 0.4], dt, goal_bubble, 0.4, 5, 0.01, "Reach Target E")
sequence.add_child(trE)

trF = WaitState(box, basket, "WaitState")
sequence.add_child(trF)


# Create Graphics
gconfig = rd.gui.Graphics.default_configuration()
gconfig.width = 1280
gconfig.height = 960
graphics = rd.gui.Graphics(gconfig)

# Create simulator object
simu = rd.RobotDARTSimu(dt)
simu.set_collision_detector("bullet")
simu.set_control_freq(100)
simu.set_graphics(graphics)
graphics.look_at((0., 4.5, 2.5), (0., 0., 0.25))
simu.add_checkerboard_floor()
simu.add_robot(robot)
simu.add_robot(box)
simu.add_robot(basket)
simu.add_visual_robot(goal_bubble)


sequence.tick_once()

episode = 0
EPISODES = 50

finished = 0

while episode < EPISODES:

    for _ in range(total_steps):

        if simu.step_world():
            break

        if simu.schedule(simu.control_freq()):
            sequence.tick_once()

    print('Episode {}/{} finished'.format(episode, EPISODES))
    episode += 1

print('Finished {}/{} of episodes'.format(finished, EPISODES))
