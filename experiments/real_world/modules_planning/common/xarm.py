XY_MIN, XY_MAX = 50, 800
X_MIN,X_MAX = 0, 800
Y_MIN,Y_MAX = -350, 350
Z_MIN, Z_MAX = 110, 650

MOVE_RATE = 1000  # hz; macro velocity = move_rate * xyz_velocity 
MOVE_SLEEP = 1 / MOVE_RATE
XYZ_VELOCITY = 1.0  # mm (in every MOVE_SLEEP interval)
ANGLE_VELOCITY_MAX = 0.05  # degree

GRIPPER_OPEN_MAX = 800
GRIPPER_OPEN_MIN = 0

POSITION_UPDATE_INTERVAL = 0.02  # 100hz the reader must read faster than this
COMMAND_CHECK_INTERVAL = 0.02  # 100hz command must be sent faster than this