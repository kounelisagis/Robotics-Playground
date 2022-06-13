import dartpy  # OSX breaks if this is imported before RobotDART
from utils import box_into_basket, damped_pseudoinverse
import numpy as np
import py_trees
from PITask import PITask
import time


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
        self.eef_link_name = "gripper_link"

        self.dt = dt
        self.arm_dofs = ["arm_1_joint", "arm_2_joint", "arm_3_joint", "arm_4_joint", "arm_5_joint", "arm_6_joint", "arm_7_joint"]
        self.finger_velocity_end = finger_velocity_end
        self.Kp = Kp
        self.Ki = Ki

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
        self.robot = robot
        self.goal = goal
        self.offset = offset
        self.goal_bubble = goal_bubble
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

        self.Kp = 0.5
        self.Ki = 0.001
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
        self.MAX_TIME = 10.

        self.logger.debug("%s.__init__()" % (self.__class__.__name__))

    def setup(self):
        self.logger.debug("%s.setup()->does nothing" % (self.__class__.__name__))

    def initialise(self):
        self.logger.debug("%s.initialise()->init controller" % (self.__class__.__name__))

        # get start time
        self.start_time = time.time()

    def update(self):
        new_status = py_trees.common.Status.RUNNING

        box_translation = self.box.base_pose().translation()
        basket_translation = self.basket.base_pose().translation()
        if box_into_basket(box_translation, basket_translation, 0.):
            new_status = py_trees.common.Status.SUCCESS
        elif time.time() - self.start_time > self.MAX_TIME:
            new_status = py_trees.common.Status.FAILURE

        if new_status == py_trees.common.Status.SUCCESS:
            self.feedback_message = "Reached target"
            self.logger.debug("%s.update()[%s->%s][%s]" % (self.__class__.__name__, self.status, new_status, self.feedback_message))
        elif new_status == py_trees.common.Status.FAILURE:
            self.feedback_message = "Failed to reach target"
            self.logger.debug("%s.update()[%s->%s][%s]" % (self.__class__.__name__, self.status, new_status, self.feedback_message))
        else:
            self.feedback_message = "Waiting"
            self.logger.debug("%s.update()[%s][%s]" % (self.__class__.__name__, self.status, self.feedback_message))
        return new_status

    def terminate(self, new_status):
        self.logger.debug("%s.terminate()[%s->%s]" % (self.__class__.__name__, self.status, new_status))
