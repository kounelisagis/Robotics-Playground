import RobotDART as rd
from utils import create_grid
import numpy as np
import py_trees
from behaviours import ReachArmTarget, ReachWheelTarget, ReachGripperTarget, WaitState


def move_box_basket_robot(basket_positions, box_positions, box_size, basket, box, robot):
    # Random box position
    box_pt = np.random.choice(len(box_positions))
    box_pose = [0., 0., 0., box_positions[box_pt][0], box_positions[box_pt][1], box_size[2] / 2.0]
    box.set_base_pose(box_pose)

    # Random basket position
    basket_pt = np.random.choice(len(basket_positions))
    basket_z_angle = 0.
    basket_pose = [0., 0., basket_z_angle, basket_positions[basket_pt][0], basket_positions[basket_pt][1], 0.0008]
    basket.set_base_pose(basket_pose)

    robot.reset()
    arm_dofs = ["arm_1_joint", "arm_2_joint", "arm_3_joint", "arm_4_joint", "arm_5_joint", "arm_6_joint", "arm_7_joint", "gripper_finger_joint", "gripper_right_finger_joint"]
    robot.set_positions(np.array([np.pi/2., np.pi/4., 0., np.pi/2., 0. , 0., np.pi/2., 0.03, 0.03]), arm_dofs)


if __name__ == '__main__':

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
    box = rd.Robot.create_box(box_size, box_pose, "free", 0.1, [0.9, 0.1, 0.1, 1.0])

    # Create basket
    basket_packages = [("basket", "models/basket")]
    basket = rd.Robot("models/basket/basket.urdf", basket_packages, "basket")
    basket.fix_to_world()

    goal_bubble = rd.Robot.create_ellipsoid(dims=[0.15, 0.15, 0.15], color=[0., 1., 0., 0.5], ellipsoid_name="target")


    # Behavior Tree
    py_trees.logging.level = py_trees.logging.Level.DEBUG

    # Create sequence node
    root = py_trees.composites.Sequence(name="Pick 'n' Place")

    pick = py_trees.composites.Sequence(name="Pick")
    place = py_trees.composites.Sequence(name="Place")

    trA = ReachWheelTarget(robot, box, -0.5, dt, goal_bubble, "Wheel Approach Cube")
    pick.add_child(trA)

    trB = ReachArmTarget(robot, box, [0., 0, 0.2], dt, goal_bubble, 5, 0.5, 1e-1, "Arm Approach Cube 1")
    pick.add_child(trB)

    trB_hat = ReachArmTarget(robot, box, [0., 0, 0.15], dt, goal_bubble, 10, 0.5, 1e-2, "Arm Approach Cube 2")
    pick.add_child(trB_hat)

    trH = ReachGripperTarget(robot, close=True, dt=dt, name="Close Gripper")
    pick.add_child(trH)

    trC = ReachArmTarget(robot, box, [0., 0, 0.4], dt, goal_bubble, 20, 0.5, 1e-2, "Pick Cube")
    pick.add_child(trC)

    trD = ReachWheelTarget(robot, basket, -0.5, dt, goal_bubble, "Wheel Approach Basket")
    place.add_child(trD)

    # trE = ReachArmTarget(robot, basket, [0., 0., 0.4], dt, goal_bubble, 5, 0.01, 1e-1, "Arm Approach Basket")
    # place.add_child(trE)

    trK = ReachGripperTarget(robot, close=False, dt=dt, name="Open Gripper")
    place.add_child(trK)

    root.add_children([pick, place, WaitState(box, basket)])


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


    EPISODES = 50
    finished = 0

    py_trees.display.render_dot_tree(root)

    for episode in range(EPISODES):

        move_box_basket_robot(basket_positions, box_positions, box_size, basket, box, robot)

        for _ in range(total_steps):
            if simu.step_world():
                break

            if simu.schedule(simu.control_freq()):
                root.tick_once()
                if root.status == py_trees.common.Status.SUCCESS:
                    print('Episode {} finished!'.format(episode+1))
                    finished += 1
                    break
                elif root.status == py_trees.common.Status.FAILURE:
                    print('Episode {} failed!'.format(episode+1))
                    break
        else:
            print('Episode {} didn\'t finish!'.format(episode+1))


    print('Finished {}/{} episodes'.format(finished, EPISODES))
