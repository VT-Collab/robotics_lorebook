from franka import Franka

franka = Franka()

conn_gripper = franka.connect(8081)

def callback(conn_gripper):
    message = str(conn_gripper.recv(2048))
    print(message)
    return

while True:
    callback(conn_gripper)