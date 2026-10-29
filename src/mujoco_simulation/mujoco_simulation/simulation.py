import rclpy
import time
import mujoco
import yaml
import mujoco.viewer
import numpy as np
from rclpy.node import Node

class T800Sim(Node):

    def __init__(self):
        super().__init__('t800_sim')
        self.model_xml = '/home/luka/faks/master_rad/simulation_environment/t800_project/resource/t800/t800.xml'
        self.model = mujoco.MjModel.from_xml_path(self.model_xml)
        self.data = mujoco.MjData(self.model)
        self.paused = False
        self.theta_offset = 0.0
        with open('/home/luka/faks/master_rad/simulation_environment/t800_project/src/mujoco_simulation/config/t800/stand_t800.yaml') as f:
            try:
                self.config = yaml.safe_load(f)
            except Exception as exc:
                raise exc
            
    def compute_control(self):
        qp = np.array(self.data.qpos[7:], dtype=float)
        qv = np.array(self.data.qvel[6:], dtype=float)

        q_des = np.concatenate([np.array(group, dtype=float) for group in self.config['target_position']])
        kp = np.concatenate([np.array(group, dtype=float) for group in self.config['kp']])
        kd = np.concatenate([np.array(group, dtype=float) for group in self.config['kd']])

        control = kp*(q_des - qp) - kd*qv
        control[0] = 0
        control[6] = 0

        return control

    def imu_process(self):
        q = self.data.sensor('imu_quaternion').data
        R = np.zeros(9)
        mujoco.mju_quat2Mat(R, q)
        R = R.reshape(3, 3)
        return np.arctan2(R[0, 2], R[2, 2]) # [-pi, pi]

    @staticmethod
    def wrap(a):
        return np.arctan2(np.sin(a), np.cos(a))

    # J00_HIP_PITCH_L \ J06_HIP_PITCH_R
    def estimate_q(self):
        theta2 = self.wrap(self.imu_process() - self.theta_offset)
        q_enc = self.data.qpos[self.model.joint('J00_HIP_PITCH_L').qposadr[0]]
        q1 = np.pi/2.0 + theta2 + q_enc
        q2 = -q_enc
        return q1, q2

    def estimate_dq(self):
        gyro_y = self.data.sensor('imu_gyro').data[1]
        dq_enc = self.data.qvel[self.model.joint('J00_HIP_PITCH_L').dofadr[0]]
        return (gyro_y + dq_enc, -dq_enc) 

    def key_callback(self, keycode):
        if chr(keycode) == ' ':
            self.paused = not self.paused

    def run_sim(self):
        with mujoco.viewer.launch_passive(self.model, self.data, key_callback=self.key_callback) as viewer:
            start = time.time()
            viewer.cam.distance = 5.0
            viewer.cam.azimuth = 200.0

            while viewer.is_running() and time.time() - start < 99999:
                step_start = time.time()

                if not self.paused:
                    self.data.ctrl = self.compute_control()
                    mujoco.mj_step(self.model, self.data)

                    print(self.estimate_q())

                viewer.sync()

                if not self.paused:
                    time_until_next_step = self.model.opt.timestep - (time.time() - step_start)
                    if time_until_next_step > 0:
                        time.sleep(time_until_next_step)
                else:
                    time.sleep(0.01)

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