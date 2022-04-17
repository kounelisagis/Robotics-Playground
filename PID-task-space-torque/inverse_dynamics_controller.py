import numpy as np
import RobotDART as rd
import dartpy  # OSX breaks if this is imported before RobotDART
from numpy import linalg as LA
from utils import damped_pseudoinverse
import matplotlib.pyplot as plt


class PIDTask:
    def __init__(self, target, dt, Kp = 10., Ki = 0.1, Kd = 0.01):
        self._target = target
        self._dt = dt
        self._Kp = Kp
        self._Ki = Ki
        self._Kd = Kd
        self._sum_error = 0
        self._prev_error = 0
    
    def set_target(self, target):
        self._target = target
    
    # function to compute error
    def error(self, tf):
        # compute error directly in world frame
        rot_error = rd.math.logMap(self._target.rotation() @ tf.rotation().T)
        lin_error = self._target.translation() - tf.translation()
        return np.r_[rot_error, lin_error]
    
    def update(self, current):
        error_in_world_frame = self.error(current)
        self._sum_error += error_in_world_frame * self._dt

        derivative = (error_in_world_frame - self._prev_error) / self._dt
        self._prev_error = error_in_world_frame

        return self._Kp*error_in_world_frame + self._Ki*self._sum_error + self._Kd*derivative, error_in_world_frame


# Load a robot
robot = rd.Iiwa()
robot.set_actuator_types("torque")
robot.set_position_enforced(True)

# Load ghost robot for visualization
robot_ghost = robot.clone_ghost()

dt = 0.001
simu = rd.RobotDARTSimu(dt)

print('Degrees of Freedom: {}'.format(robot.num_dofs()))

world_joint, dofs, ee_joint = np.split(robot.joint_names(), [1, -1])

intial_positions = robot.positions()+np.random.rand(robot.num_dofs())*np.pi/1.5-np.pi/3.
target_positions = robot.positions()+np.random.rand(robot.num_dofs())*np.pi/1.5-np.pi/3.

# Create Graphics
gconfig = rd.gui.GraphicsConfiguration(1024, 768) # Create a window of 1024x768 resolution/size
graphics = rd.gui.Graphics(gconfig) # create graphics object with configuration
simu.set_graphics(graphics)
graphics.look_at([3., 1., 2.], [0., 0., 0.])
# graphics.record_video("inverse_dynamics_controller.mp4")

# Add robot and nice floor
robot.set_positions(intial_positions)
simu.add_robot(robot)
robot_ghost.set_positions(target_positions)
simu.add_robot(robot_ghost)
simu.add_checkerboard_floor()

# get end-effector pose
eef_link_name = "iiwa_link_ee"
tf_desired = robot_ghost.body_pose(eef_link_name)


Kp = 3.; Ki = 2.; Kd = 2.
controller = PIDTask(tf_desired, dt, Kp, Ki, Kd)

errors = []
error = 1

# Run simulation
while error > 1e-2:

    if simu.step_world():
        break
    tf = robot.body_pose(eef_link_name)
    PID, error = controller.update(tf)
    jac = robot.jacobian(eef_link_name) # this is in world frame
    jac_pinv = damped_pseudoinverse(jac)
    cmd = robot.mass_matrix(dofs) @ (jac_pinv @ PID) + robot.coriolis_gravity_forces(dofs)

    robot.set_commands(cmd)

    error = LA.norm(error)
    errors.append(error)
    print('Error:', error)


plt.plot(errors, '-')
plt.title("Inverse Dynamics Controller | Kp = {}, Ki = {}, Kd = {}".format(Kp, Ki, Kd))
plt.xlabel("timestep")
plt.ylabel("error")
plt.show()
