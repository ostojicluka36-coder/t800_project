import rclpy
import time
import mujoco
import mujoco.viewer
from rclpy.node import Node
from std_msgs.msg import String

class T800Sim(Node):

    def __init__(self):
        super().__init__('t800_sim')
        self.model_xml = '/home/luka/faks/master_rad/simulation_environment/t800_project/resource/t800/t800.xml'
        self.model = mujoco.MjModel.from_xml_path(self.model_xml)
        self.data = mujoco.MjData(self.model)

    def run_sim(self):
        with mujoco.viewer.launch_passive(self.model, self.data) as viewer:
            start = time.time()
            while viewer.is_running() and time.time() - start < 99999:
                step_start = time.time()

                mujoco.mj_step(self.model, self.data)

                viewer.sync()

                time_until_next_step = self.model.opt.timestep - (time.time() - step_start)
                if time_until_next_step > 0:
                    time.sleep(time_until_next_step)

def main():
    rclpy.init()

    node = T800Sim()
    node.run_sim()

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        node.get_logger().info('Shutting down...')
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()