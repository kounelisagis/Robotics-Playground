# Robotics-Playground

###  Quadruped Robot 
In this project, I created a urdf model for a quadruped robot with at least three degrees of freedom at each leg. I determined the sizes of the body segments and selected appropriate joint types. I loaded the model in RobotDART and wrote code to calculate the position of each body segment in space, given the positions of the joints. I used numpy and Eigen for mathematical computations.

### PID task spacetorque 
In this project, I implemented a PID controller operating in the task-space for controlling a KUKA Iiwa 14 robotic arm using torque motors. I used RobotDART to compute the necessary dynamic equations and created a scenario where the robot can move from a random initial state within joint limits to a specified target position and orientation.

### Tiago pick and place
In this project, I have successfully designed and implemented a versatile controller for the Tiago robot (PAL Robotics), enabling it to efficiently solve a "Pick and Place" scenario. This controller seamlessly allows the robot to grasp a cube from any initial position (within limits) and accurately place it into a basket, which can be positioned anywhere within the workspace. The controller is a Behavior Tree, a fusion of low-level task space controllers.
