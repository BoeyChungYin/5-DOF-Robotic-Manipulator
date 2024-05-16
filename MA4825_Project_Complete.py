import os
import time
import threading
import torch
import cv2
import argparse
import supervision as sv
import numpy as np
from ultralytics import YOLO
np.set_printoptions(suppress=True)

if os.name == 'nt':
    import msvcrt
    def getch():
        return msvcrt.getch().decode()
else:
    import sys, tty, termios
    fd = sys.stdin.fileno()
    old_settings = termios.tcgetattr(fd)
    def getch():
        try:
            tty.setraw(sys.stdin.fileno())
            ch = sys.stdin.read(1)
        finally:
            termios.tcsetattr(fd, termios.TCSADRAIN, old_settings)
        return ch

from dynamixel_sdk import *                    # Uses Dynamixel SDK library

class HomogenousMatrix3D:
    def __init__(self):
        self.rotation_matrix = np.identity(4)  # Initialize with an identity matrix

    def rotate_x(self, angle_x):
        rotation_matrix_x = np.array([[1, 0, 0, 0],
                                      [0, np.cos(angle_x), -np.sin(angle_x), 0],
                                      [0, np.sin(angle_x), np.cos(angle_x), 0],
                                      [0, 0, 0, 1]])
        self.rotation_matrix = np.matmul(self.rotation_matrix, rotation_matrix_x)

    def rotate_y(self, angle_y):
        rotation_matrix_y = np.array([[np.cos(angle_y), 0, np.sin(angle_y),0],
                                      [0, 1, 0, 0],
                                      [-np.sin(angle_y), 0, np.cos(angle_y),0],
                                      [0, 0, 0, 1]])
        self.rotation_matrix = np.matmul(self.rotation_matrix, rotation_matrix_y)

    def rotate_z(self, angle_z):
        rotation_matrix_z = np.array([[np.cos(angle_z), -np.sin(angle_z), 0, 0],
                                      [np.sin(angle_z), np.cos(angle_z), 0, 0],
                                      [0, 0, 1, 0],
                                      [0, 0, 0, 1]])
        self.rotation_matrix = np.matmul(self.rotation_matrix, rotation_matrix_z)

    def translate_x(self, displacement_x):
        translation_matrix_x = np.array([[1, 0, 0, displacement_x],
                                      [0, 1, 0, 0],
                                      [0, 0, 1, 0],
                                      [0, 0, 0, 1]])
        self.rotation_matrix = np.matmul(self.rotation_matrix, translation_matrix_x)

    def translate_y(self, displacement_y):
        translation_matrix_y = np.array([[1, 0, 0, 0],
                                      [0, 1, 0, displacement_y],
                                      [0, 0, 1, 0],
                                      [0, 0, 0, 1]])
        self.rotation_matrix = np.matmul(self.rotation_matrix, translation_matrix_y)

    def translate_z(self, displacement_z):
        translation_matrix_z = np.array([[1, 0, 0, 0],
                                      [0, 1, 0, 0],
                                      [0, 0, 1, displacement_z],
                                      [0, 0, 0, 1]])
        self.rotation_matrix = np.matmul(self.rotation_matrix, translation_matrix_z)

    def get_matrix(self):
        return self.rotation_matrix
    
    def print(self,message=""):
        temp=self.rotation_matrix
        print(message)
        np.set_printoptions(precision=3)
        np.set_printoptions(suppress=True)
        print(temp)

ADDR_MX_TORQUE_ENABLE      = 24               # Control table address is different in Dynamixel model
ADDR_MX_GOAL_POSITION      = 30
ADDR_MX_q_list             = 36
ADDR_MX_MOVING_SPEED       = 32
ADDR_MX_PRESENT_SPEED      = 38

# Protocol version
PROTOCOL_VERSION           = 1.0               # See which protocol version is used in the Dynamixel

# Default setting
# Dynamixel#1 ID : 1
# Dynamixel#2 ID : 2
# Dynamixel#3 ID : 3
# Dynamixel#4 ID : 4
# Dynamixel#5 ID : 5
# Dynamixel#6 ID : 6

BAUDRATE                   = 57600             # Dynamixel default baudrate : 57600
DEVICENAME1                = 'COM5'            # Check which port is being used on your controller 'COM5'

TORQUE_ENABLE              = 1                 # Value for enabling the torque
TORQUE_DISABLE             = 0                 # Value for disabling the torque

# Initialize PortHandler instance
# Set the port path
# Get methods and members of PortHandlerLinux or PortHandlerWindows
portHandler1 = PortHandler(DEVICENAME1)
print("PORTHANDLER: ",portHandler1)

# Initialize PacketHandler instance
# Set the protocol version
# Get methods and members of Protocol1PacketHandler or Protocol2PacketHandler
packetHandler = PacketHandler(PROTOCOL_VERSION)

# Open port1
if portHandler1.openPort():
    print("Succeeded to open the port")
else:
    print("Failed to open the port")
    print("Press any key to terminate...")
    getch()
    quit()

# Set port1 baudrate
if portHandler1.setBaudRate(BAUDRATE):
    print("Succeeded to change the baudrate")
else:
    print("Failed to change the baudrate")
    print("Press any key to terminate...")
    getch()
    quit()

# Change webcam resolution
def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Welding Camera")
    parser.add_argument(
        "--webcam-resolution", 
        default=[1280, 720], 
        nargs=2, 
        type=int)
    args = parser.parse_args()
    return args

# YOLO Processing
def yolo(pframe):
    model = YOLO("yolov8n_model.pt") # "yolov8n_model.pt" is trained model name

    result = model(pframe, agnostic_nms=True, save=False, conf=0.7, device=0, max_det=1)[0] # Obtain results from detection using YOLO Model
    detections = sv.Detections.from_ultralytics(result)

    item_list = ["Block", "T-Junction"] # To label objects where in YOLO, 0 is Block, 1 is T-Junction
    obj = "NA" # Initialize object as NA
    # Only consider object detected as valid if the center of the object in within a set range
    for item in detections:
        cords = [round(x) for x in item[0]]
        x_center = cords[0] + ((cords[2] - cords[0])/2)
        y_center = cords[1] + ((cords[3] - cords[1])/2)
        center = [x_center, y_center]
        center = [round(x) for x in center]
        if center[0] in range(320,960) and center[1] in range(180,540):
            index = int(item[3])
            obj = str(item_list[index])
    return(obj)

# Process Frame to Overlap with Video Feed
def label(pframe):
    global frame, obj
    if phase == 0:
        obj = yolo(pframe)
    image = cv2.rectangle(pframe, (0,0), (700,30), (0,0,0), -1)
    frame = cv2.putText(image, text, (0,25), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255,255,255), 2)

# Camera running Parallel to Main Program
def feed():
    global start
    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, frame_width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, frame_height)
    while True:
        ret, pframe = cap.read()
        label(pframe)
        cv2.imshow('Welding Camera', frame)
        # 30 miliseconds opencv waits for us to hit keys on keyboard, 27 is esc in ascii
        key_pressed = cv2.waitKey(30)
        if key_pressed == 27:
            start = -1
            break
        elif key_pressed != -1:
            start = 1

    cap.release()
    cv2.destroyAllWindows()

# Enable Dynamixel#1 Torque
# FUNCTION FOR EACH MOTOR TO ENABLE TORQUE
def enabledynamixeltorque(packetHandler,portHandler1, DXL1_ID, ADDR_MX_TORQUE_ENABLE, TORQUE_ENABLE):
    dxl_comm_result, dxl_error = packetHandler.write1ByteTxRx(portHandler1, DXL1_ID, ADDR_MX_TORQUE_ENABLE, TORQUE_ENABLE)
    if dxl_comm_result != COMM_SUCCESS:
        print("%s" % packetHandler.getTxRxResult(dxl_comm_result))
        print("\terror: packetHandler.getTxRxResult(dxl_comm_result)",DXL1_ID)
    elif dxl_error != 0:
        print("%s" % packetHandler.getRxPacketError(dxl_error))
        print("\terror: packetHandler.getRxPacketError(dxl_error)",DXL1_ID)
    else:
        print("Dynamixel#%d has been successfully connected" % DXL1_ID)
    return dxl_comm_result, dxl_error

# Enable Torque for All Motors
for id in range(1,7):
    enabledynamixeltorque(packetHandler, portHandler1, id, ADDR_MX_TORQUE_ENABLE, TORQUE_ENABLE)

# Write Goal Position
def writedynamixelgoalposition(packetHandler,portHandler1, DXL1_ID, ADDR_MX_GOAL_POSITION, dxl_goal_position):
    dxl_comm_result, dxl_error = packetHandler.write2ByteTxRx(portHandler1, DXL1_ID, ADDR_MX_GOAL_POSITION, dxl_goal_position)
    if dxl_comm_result != COMM_SUCCESS:
        print("%s" % packetHandler.getTxRxResult(dxl_comm_result))
    elif dxl_error != 0:
        print("%s" % packetHandler.getRxPacketError(dxl_error))
    return dxl_comm_result, dxl_error

# Read Present Position
def readdynamixelpresentposition(packetHandler,portHandler1, DXL1_ID, ADDR_MX_q_list):
    dxl1_q_list, dxl_comm_result, dxl_error = packetHandler.read2ByteTxRx(portHandler1, DXL1_ID, ADDR_MX_q_list)
    if dxl_comm_result != COMM_SUCCESS:
        print(packetHandler.getTxRxResult(dxl_comm_result))
    elif dxl_error != 0:
        print(packetHandler.getRxPacketError(dxl_error))
    return dxl1_q_list, dxl_comm_result, dxl_error

# Write Speed
def writedynamixelspeed(packetHandler,portHandler, DXL_ID, ADDR_MX_MOVING_SPEED, DXL_MOVING_SPEED):
    DXL_MOVING_SPEED = round(DXL_MOVING_SPEED) # Round speed to whole number
    if DXL_MOVING_SPEED < 10: 
        DXL_MOVING_SPEED = 10 # Set minimum speed of 10
    dxl_comm_result, dxl_error = packetHandler.write2ByteTxRx(portHandler, DXL_ID, ADDR_MX_MOVING_SPEED, DXL_MOVING_SPEED)
    if dxl_comm_result != COMM_SUCCESS:
        print("%s" % packetHandler.getTxRxResult(dxl_comm_result))
    elif dxl_error != 0:
        print("%s" % packetHandler.getRxPacketError(dxl_error))
    return dxl_comm_result, dxl_error

# Read Moving Speed
def readdynamixelmovingspeed(packetHandler,portHandler, DXL_ID, ADDR_MX_MOVING_SPEED):
    dxl_speed, dxl_comm_result, dxl_error = packetHandler.read2ByteTxRx(portHandler, DXL_ID, ADDR_MX_MOVING_SPEED)
    if dxl_comm_result != COMM_SUCCESS:
        print("%s" % packetHandler.getTxRxResult(dxl_comm_result))
    elif dxl_error != 0:
        print("%s" % packetHandler.getRxPacketError(dxl_error))
    return dxl_speed, dxl_comm_result, dxl_error

# Read Present Speed
def readdynamixelpresentspeed(packetHandler,portHandler1, DXL_ID, ADDR_MX_PRESENT_SPEED):
    dxl_speed, dxl_comm_result, dxl_error = packetHandler.read2ByteTxRx(portHandler1, DXL_ID, ADDR_MX_PRESENT_SPEED)
    if dxl_comm_result != COMM_SUCCESS:
        print("%s" % packetHandler.getTxRxResult(dxl_comm_result))
    elif dxl_error != 0:
        print("%s" % packetHandler.getRxPacketError(dxl_error))
    return dxl_speed, dxl_comm_result, dxl_error

# Find Goal Position for Each Servo
def position(dxl_goal_xyz):
    x = dxl_goal_xyz[0]
    y = dxl_goal_xyz[1]
    z = dxl_goal_xyz[2] + 20 # Added +20mm offset to not hit the workpiece
    dxl_goal_rad = [0,0,0,0,0,0]
    dxl_goal_pwm = []

    # Finding Goal Position For Each Joint
    r1 = np.sqrt(x**2 + y**2)
    r2 = np.sqrt(r1**2 + (z+41)**2)

    dxl_goal_rad[0] = np.arctan2(y,x)                                           # theta 1

    dxl_goal_rad[1] = (np.pi/2) - np.arcsin((z+41)/r2) - np.arccos(r2/(2*93.5)) # theta 2

    dxl_goal_rad[2] = -(np.arccos(1-((r2**2)/(2*(93.5**2)))) - (np.pi/2))       # theta 3

    dxl_goal_rad[3] = -(dxl_goal_rad[1] + dxl_goal_rad[2] + np.deg2rad(3))      # theta 4

    dxl_goal_rad[4] = dxl_goal_rad[0] + tool_count*(np.pi/2)                    # theta 5

    dxl_goal_rad[5] = -(np.pi/4)                                                # theta 6 (-45 degrees)

    # Convert list from Radian to Deg
    dxl_goal_deg = np.rad2deg(dxl_goal_rad)

    # Ensure Within Bounds
    for i in range(0,6):
        if dxl_goal_deg[i] > 180:
            dxl_goal_deg[i] = dxl_goal_deg[i] - 360
        elif dxl_goal_deg[i] < -180:
            dxl_goal_deg[i] = dxl_goal_deg[i] + 360
        goal_pwm = (512/150)*dxl_goal_deg[i] + 512
        if goal_pwm > 1023:
            goal_pwm = 1023
        elif goal_pwm < 0:
            goal_pwm = 0
        dxl_goal_pwm.append(round(goal_pwm))

    return dxl_goal_pwm

# Sets Velocity
def velocity(joint_vel):
    for i in range(1,7):
        dxl_comm_result, dxl_error = writedynamixelspeed(packetHandler,portHandler1, i, ADDR_MX_MOVING_SPEED, joint_vel[i-1])

# Forward Kinematics
def FK(q_pos_pwm): 

    # Convert the qi values from pwm to degree
    qi = []
    for i in range(0,5):
        q_pos = (q_pos_pwm[i]-512)*(150/512)
        qi.append(q_pos)
    qi = np.rad2deg(qi)

    # Assume q6 is locked, omit q6
    q1, q2, q3, q4, q5 = qi # Expect qi to be 5 element list
    he = 0 #initialize variable for 4x4 matrix
    
    #initialize 4x4 homogenous matrix
    matrix = HomogenousMatrix3D()
    
    #global reference frame to motor 1: LINK displacement
    matrix.translate_z(90) 

    #motor 1: ANGLE #q1=0
    matrix.rotate_z(np.radians(q1)) 

    #motor 2: ANGLE #q2=0
    matrix.rotate_y(np.radians(q2))

    #motor 2 to motor 3: LINK displacement
    matrix.translate_z(93.5) 

    #motor 3: ANGLE q3=0
    matrix.rotate_y(np.radians(q3)) 

    #motor 3 to motor 4: LINK displacement
    matrix.translate_x(93.5) 

    #motor 4: ANGLE q4=0
    matrix.rotate_y(np.radians(q4)) 

    #motor 5: ANGLE q5=0
    matrix.rotate_z(np.radians(q5)) 

    #motor 5 to end effector. (assumme motor 6 locked)
    matrix.translate_z(-131)
    
    #homogenous end effector 4x4 matrix
    he = matrix.get_matrix()
         
    #returns translation vectorr
    return he[:3,3]

# Segment points further
def pt_maker(goal, start, diff):
    n = 5 # No. of divisions
    set_goal_list = []
    goal_interval = np.dot(1/n, diff)
    
    for i in range(0,n):
        set_goal = start + np.dot(i, goal_interval)
        set_goal = set_goal.tolist()
        set_goal_list.append(set_goal)
    return set_goal_list

# Checks if more points are needed
def goal_maker(goal):
    dist = 30 # Criteria to divide to more pts
    final_list = []
    for a in range(0, len(goal)-1):
        diff = np.subtract(goal[a+1], goal[a])
        if max(abs(diff)) >= dist:
            set_goal_list = pt_maker(goal[a+1], goal[a], diff)
            for item in set_goal_list:
                final_list.append(item)
        else:
            final_list.append(goal[a])
    final_list.append(goal[-1])
    return final_list

# Set constant rotational speed for each motor between points
def const_spd_maker(dxl_goal_pwm, q_pos_pwm):
    t = 1.5
    pwm_diff = np.subtract(dxl_goal_pwm, q_pos_pwm)
    pwm_diff = abs(pwm_diff)
    pwm_spd = np.dot(1/t, pwm_diff)
    return pwm_spd

global tool_count, text, phase                  # Global count to multiply by to offset theta 5
tool_count, phase, start = 0, 0, 0              # Initial theta 5 is 0
text = "Setting Up"                             # Initialization Text, Text to be overlayed onto raw camera feed                                    
tol = 15                                        # Tolerance for each servo

# Return to Home
joint_vel = [40, 40, 40, 40, 40, 40]            # Initial Startup Speed
velocity(joint_vel)                             # Set Velocity
dxl_goal_home = [93.5, 0, 52.5]                 # Goal position for Home In XYZ [93.5, 0, 52.5]
dxl_goal_pwm = position(dxl_goal_home)          # Set Goal Position

# Initialisation before start of Point to Point
args = parse_arguments()
frame_width, frame_height = args.webcam_resolution

# Paths Set
# T path for T-junction, B path for Block
T = [[100.5, -12.5, 20.0], [100.5, -5, 10.0], [130.5, -5, 10.0], [160.5, -5, 10.0], [165.5, -12.5, 20.0], [165.5, 12.5, 20.0], [160.5, 5, 10.0], [100.5, 5, 10.0], [100.5, -12.5, 20.0]]
B = [[105.5, 25.0, 20.0], [120.5, 15.0, 10.0], [120.5, -15.0, 10.0], [105.5, -25.0, 20.0], [115.5, -20.0, 10.0], [145.5, -20.0, 10.0], [155.5, -25.0, 20.0], [155.5, -20.0, 10.0], [155.5, 15.0, 10.0], [155.5, 25.0, 20.0], [145.5, 15.0, 10.0], [115.5, 15.0, 10.0], [105.5, 25.0, 20.0], [105.5, 25.0, 50.0]]
time.sleep(1)

# Start parallel processing for camera
thread = threading.Thread(target=feed)
thread.start()

# Start up time to not get error from missing obj variable
time.sleep(15) 

while True:
    n, phase, start = 0, 0, 0      # Point to Run in dxl_goal_final
    text = "Press Any Key to Start! (or press ESC to Quit!)"
    while start == 0:
        print("Waiting for Input")

    if start == -1:
        break

    # Detection Phase
    while obj == "NA":
        text = "Detecting Object."
        time.sleep(1)
        text = "Detecting Object.."
        time.sleep(1)
        text = "Detecting Object..."
        time.sleep(1)

    phase = 1 # Cuts off object detection
    lobj = obj # Locks obj just in case it changes

    # Transitioning
    text = "Detected " + str(lobj)
    if lobj == "Block":
        dxl_goal_xyz = B
    elif lobj == "T-Junction":
        dxl_goal_xyz = T
    else:
        dxl_goal_xyz = ["NA"]
    time.sleep(3)
   
    dxl_goal_final = goal_maker(dxl_goal_xyz)       # Construct new list to run using dxl_goal_xyz

    while True:
        # Run to Next Goal in List If Not End of List
        if n == (len(dxl_goal_final)-1):
            break

        goal_pos = dxl_goal_final[n]               # Set Current Goal Position to Reach for this Loop in XYZ
        text = "Moving to Point:" + str(goal_pos)
        #print(goal_pos)
        if goal_pos == [165.5, 12.5, 20.0]:        # Turn tool by 180 degrees
            tool_count += 2
        elif goal_pos == [105.5, 25.0, 20.0]:      # Turn tool by 90 degrees
            tool_count += 1
        elif goal_pos == [105.5, -25.0, 20.0] or goal_pos == [155.5, 25.0, 20.0] or goal_pos == [155.5, -25.0, 20.0]: # Turn tool by -90 degrees
            tool_count -= 1

        # Read Current Position for all Servos
        q_pos_pwm = []                             # Set Empty List to Store All Theta Values
        for i in range(1,7):
            dxl_q, dxl_comm_result, dxl_error = readdynamixelpresentposition(packetHandler,portHandler1, i, ADDR_MX_q_list)
            q_pos_pwm.append(dxl_q)

        # Forward Kinematics
        dxl_curr_xyz = FK(q_pos_pwm[0:5])

        # Find Next Goal Position for Each Servo
        dxl_goal_pwm = position(goal_pos)

        # Find Next Speed to Set for Each Servo
        pwm_spd = const_spd_maker(dxl_goal_pwm, q_pos_pwm)

        # Set Speed To Run for Each Servo
        velocity(pwm_spd)

        # Set Goal Position for Each Servo
        for i in range(1,7):
            dxl_comm_result, dxl_error = writedynamixelgoalposition(packetHandler,portHandler1, i, ADDR_MX_GOAL_POSITION, dxl_goal_pwm[i-1])

        # Set Goal Position in terms of theta of each joint
        while True:
            # Current q for All Motors
            q_pos_pwm= []                         # Set Empty List to Store All Theta Values and Speed
            for i in range(1,7):
                dxl_q, dxl_comm_result, dxl_error = readdynamixelpresentposition(packetHandler,portHandler1, i, ADDR_MX_q_list)
                q_pos_pwm.append(dxl_q)

            reached = []
            for i in range(0,6):
                reach = (abs(dxl_goal_pwm[i] - q_pos_pwm[i]) < tol)
                reached.append(reach)
            print("Reached: ", reached)
            if all(reached) == True:
                print("Current XYZ: ", dxl_curr_xyz)
                break

        n += 1 # Move On to Next Goal

    # Run End
    text = "Run Finished"

    time.sleep(3)

    # Return to Home
    joint_vel = [40, 40, 40, 40, 40, 40]           # Initial Startup Speed
    velocity(joint_vel)                            # Set Velocity
    dxl_goal_home = [93.5, 0, 52.5]                # Goal position for Home In XYZ [93.5, 0, 52.5]
    dxl_goal_pwm = position(dxl_goal_home)         # Set Goal Position
    time.sleep(5)

    tool_count = 0                                 # Return Tool to Home Position
    dxl_goal_pwm = position(dxl_goal_home)         # Set Goal Position

text = "Exiting Program"

# Disable Dynamixel#1 Torque
def disabledynamixeltorque(packetHandler,portHandler1, DXL1_ID, ADDR_MX_TORQUE_ENABLE, TORQUE_DISABLE):
    dxl_comm_result, dxl_error = packetHandler.write1ByteTxRx(portHandler1, DXL1_ID, ADDR_MX_TORQUE_ENABLE, TORQUE_DISABLE)
    if dxl_comm_result != COMM_SUCCESS:
        print("%s" % packetHandler.getTxRxResult(dxl_comm_result))
    elif dxl_error != 0:
        print("%s" % packetHandler.getRxPacketError(dxl_error))

for i in range(1,7):
    disabledynamixeltorque(packetHandler,portHandler1, i, ADDR_MX_TORQUE_ENABLE, TORQUE_DISABLE)

# Close port1
portHandler1.closePort()

