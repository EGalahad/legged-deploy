
import time
import datetime
import numpy as np
import math
import torch
import itertools
import h5py
import argparse
import os

from setproctitle import setproctitle
from scipy.spatial.transform import Rotation as R
from go2deploy import ONNXModule, init_channel, RobotIface
from go2deploy.utils import lerp, normalize
from torch.utils._pytree import tree_map

try:
    from tensordict import TensorDict
    from torchrl.envs.utils import set_exploration_type, ExplorationType
except ModuleNotFoundError:
    pass

np.set_printoptions(precision=3, suppress=True, floatmode="fixed")

ORBIT_JOINT_ORDER = [
    'FL_hip_joint', 'FR_hip_joint', 'RL_hip_joint', 'RR_hip_joint', 
    'FL_thigh_joint', 'FR_thigh_joint', 'RL_thigh_joint', 'RR_thigh_joint', 
    'FL_calf_joint', 'FR_calf_joint', 'RL_calf_joint', 'RR_calf_joint'
]

SDK_JOINT_ORDER = [
    'FR_hip_joint', 'FR_thigh_joint', 'FR_calf_joint',
    'FL_hip_joint', 'FL_thigh_joint', 'FL_calf_joint',
    'RR_hip_joint', 'RR_thigh_joint', 'RR_calf_joint',
    'RL_hip_joint', 'RL_thigh_joint', 'RL_calf_joint'
]


class Go2Iface:

    smoothing_length: int = 5
    smoothing_ratio: float = 0.4
    command_dim: int

    def __init__(self, cfg, log_file: h5py.File=None):
        self.cfg = cfg
        self.log_file = log_file

        self.acc_bias = np.array([0.851, 0.310, 9.580])

        self._robot = RobotIface()
        self._robot.start_control(interval=2000)
        self.default_joint_pos = np.array(
            [
                0.1, -0.1,  0.1, -0.1,  
                0.78,  0.78,  0.75,  0.75, 
                -1.5, -1.5, -1.5, -1.5
            ], 
        )
        
        self.dt = 0.02
        self.latency = 0.0
        self.rpy = np.zeros(3)
        self.angvel_history = np.zeros((3, self.smoothing_length))
        self.projected_gravity_history = np.zeros((3, self.smoothing_length))
        self.angvel = np.zeros(3)
        self.action_buf = np.zeros((12, 4))
        self.command = np.zeros(self.command_dim, dtype=np.float32)
        self.lxy = 0.
        self.rxy = 0.
        self.action_buf_steps = 1
        self.last_action = np.zeros(12)
        self.start_t = time.perf_counter()
        self.timestamp = time.perf_counter()
        self.step_count = 0

        # self.kalman_filter = KalmanFilter3D()
        self.update_state()
        self.update_command()
        _obs = self._compute_obs()
        self.obs_dim = _obs.shape[0]
        self.obs_buf = np.zeros((self.obs_dim * 6))
        
        if self.log_file is not None:
            default_len = 50 * 60
            self.log_file.attrs["cursor"] = 0
            log_file.create_dataset("observation", (default_len, self.obs_dim), maxshape=(None, self.obs_dim))
            log_file.create_dataset("action", (default_len, 12), maxshape=(None, 12))

            log_file.create_dataset("rpy", (default_len, 3), maxshape=(None, 3))
            log_file.create_dataset("jpos", (default_len, 12), maxshape=(None, 12))
            log_file.create_dataset("jvel", (default_len, 12), maxshape=(None, 12))
            log_file.create_dataset("jpos_des", (default_len, 12), maxshape=(None, 12))
            log_file.create_dataset("tau_est", (default_len, 12), maxshape=(None, 12))
            log_file.create_dataset("quat", (default_len, 4), maxshape=(None, 4))
            log_file.create_dataset("linvel", (default_len, 3), maxshape=(None, 3))
            log_file.create_dataset("angvel", (default_len, 3), maxshape=(None, 3))
    
    def reset(self):
        self.start_t = time.perf_counter()
        self.update_state()
        self.update_command()
        return self._compute_obs()
    
    def update_state(self):
        self.robot_state = self._robot.get_robot_state()

        self.prev_rpy = self.rpy
        # self.rpy = self._robot.get_rpy()
        (
            self.jpos_sdk, self.jvel_sdk, self.tau_sdk,
            self.rpy, angvel, self.feet_force
        ) = np.split(self._robot.get_full_state(), [12, 24, 36, 39, 42])
        
        self.jpos_sim = self.sdk_to_orbit(self.jpos_sdk)
        self.jvel_sim = self.sdk_to_orbit(self.jvel_sdk)
        self.tau_sim = self.sdk_to_orbit(self.tau_sdk)

        dt = time.perf_counter() - self.timestamp
        
        # angvel = ((self.rpy - self.prev_rpy) / dt).clip(-3, 3)
        self.angvel_history = np.roll(self.angvel_history, -1, axis=1)
        self.angvel_history[:, -1] = angvel
        self.angvel = lerp(self.angvel, self.angvel_history.mean(axis=1), self.smoothing_ratio)
        
        self.projected_gravity_history[:, 1:] = self.projected_gravity_history[:, :-1]
        self.projected_gravity_history[:, 0] = self._robot.get_projected_gravity()
        self.projected_gravity = normalize(self.projected_gravity_history.mean(1))

        self.lxy = lerp(self.lxy, self._robot.lxy(), 0.5)
        self.rxy = lerp(self.rxy, self._robot.rxy(), 0.5)
        # self.latency = (datetime.datetime.now() - self._robot.timestamp).total_seconds()

    
    def apply_action(self, action: np.ndarray):
        self.action_buf[:, 1:] = self.action_buf[:, :-1]
        self.action_buf[:, 0] = action.clip(-6, 6)

        self.last_action = self.last_action * 0.2 + self.action_buf[:, 0] * 0.8
        jpos_target = self.last_action * 0.5 + self.default_joint_pos
        jpos_target = jpos_target.clip(-np.pi, np.pi)
        self.jpos_target = jpos_target
        self._robot.set_command(self.orbit_to_sdk(jpos_target))
        self.step_count += 1

    def get_obs(self):
        self.update_state()
        self.update_command()
        self._maybe_log()
        obs = self._compute_obs()
        # self.obs_buf = np.roll(self.obs_buf, -self.obs_dim)
        # self.obs_buf[-self.obs_dim:] = obs
        return obs
        
    def update_command(self):
        pass

    def _compute_obs(self):
        raise NotImplementedError
    
    def _maybe_log(self):
        if self.log_file is None:
            return
        self.log_file["action"][self.step_count] = self.action_buf[:, 0]
        self.log_file["angvel"][self.step_count] = self.angvel
        self.log_file["linvel"][self.step_count] = self._robot.get_velocity()
        self.log_file["rpy"][self.step_count] = self.rpy
        self.log_file["jpos"][self.step_count] = self.jpos_sim
        self.log_file["jvel"][self.step_count] = self.jvel_sim
        self.log_file["jpos_des"][self.step_count] = self.jpos_target
        self.log_file["tau_est"][self.step_count] = self.tau_sim
        self.log_file.attrs["cursor"] = self.step_count

        if self.step_count == self.log_file["jpos"].len() - 1:
            new_len = self.step_count + 1 + 3000
            print(f"Extend log size to {new_len}.")
            for key, value in self.log_file.items():
                value.resize((new_len, value.shape[1]))

    @staticmethod
    def orbit_to_sdk(joints: np.ndarray):
        return np.flip(joints.reshape(3, 2, 2), axis=2).transpose(1, 2, 0).reshape(-1)
    
    @staticmethod
    def sdk_to_orbit(joints: np.ndarray):
        return np.flip(joints.reshape(2, 2, 3), axis=1).transpose(2, 0, 1).reshape(-1)

    def process_action(self, action: np.ndarray):
        return self.orbit_to_sdk(action * 0.5 + self.default_joint_pos)
    
    def process_action_inv(self, jpos_sdk: np.ndarray):
        return (self.sdk_to_orbit(jpos_sdk) - self.default_joint_pos) / 0.5



class Go2Vel(Go2Iface):
    
    command_dim: int = 4 # linvel_xy, angvel_z, base_height

    def update_command(self):
        t = time.perf_counter() - self.start_t
        vx = np.sin(t * 0.75)
        self.command[0] = lerp(self.command[0] * 0.95, self.lxy[1] * 2.0, 0.2)
        self.command[1] = lerp(self.command[1], -self.lxy[0], 0.2)

        self.command[2] = -self.rxy[0] * 1.2
        self.command[3] = 0.75 #

    def _compute_obs(self):
        # self.rot = R.from_euler("xyz", self.rpy)
        # angvel = self.rot.inv().apply(self.angvel)
        angvel = self.angvel
        
        obs = [
            self.command,
            # angvel,
            self.projected_gravity,
            self.jpos_sim,
            self.jvel_sim,
            self.action_buf[:, :self.action_buf_steps].reshape(-1),
        ]
        obs = np.concatenate(obs, dtype=np.float32)
        return obs


class Go2Impd(Go2Iface):

    command_dim: int = 10 # setpose_xy, setpose_yaw, kp_xy, kp_yaw, kd_xyz, vmass

    def update_command(self):
        kp = (self.rxy[1] + 1) / 2 * (10 - 2) + 2
        kd = 2 * math.sqrt(kp)
        self.command[0] = 0.5 # self.lxy[1]
        self.command[1] = 0.0
        self.command[2] = 0.0
        self.command[3:5] = self.command[:2] * kp
        self.command[5:6] = kd
        self.command[6:7] = kp
        self.command[7:8] = kd
        self.command[8:9] = 3.0

    def _compute_obs(self):
        # self.rot = R.from_euler("xyz", self.rpy)
        # angvel = self.rot.inv().apply(self.angvel)
        angvel = self.angvel
        
        obs = [
            # angvel,
            self.projected_gravity,
            self.jpos_sim,
            self.jvel_sim,
            self.action_buf[:, :self.action_buf_steps].reshape(-1),
        ]
        obs = np.concatenate(obs, dtype=np.float32)
        return obs


@torch.inference_mode()
@set_exploration_type(ExplorationType.MODE)
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-p", "--path", type=str)
    parser.add_argument("-l", "--log", action="store_true", default=False)
    args = parser.parse_args()

    timestr = datetime.datetime.now().strftime("%m-%d_%H-%M-%S")
    setproctitle("play_go2")

    init_channel("enp58s0")

    init_pos = np.array([
        0.0, 0.9, -1.8, 
        0.0, 0.9, -1.8, 
        0.0, 0.9, -1.8, 
        0.0, 0.9, -1.8
    ])

    if args.log:
        os.makedirs("logs", exist_ok=True)
        log_file = h5py.File(f"logs/{timestr}.h5py", "a")
    else:
        log_file = None

    robot = Go2Impd({}, log_file)
    
    robot._robot.set_kp(25.)
    robot._robot.set_kd(0.5)

    path = args.path
    if path.endswith(".onnx"):
        backend = "onnx"
        policy_module = ONNXModule(path)
        def policy(inp):
            out = policy_module(inp)
            action = out["action"].reshape(-1)
            carry = {k[1]: v for k, v in out.items() if k[0] == "next"}
            return action, carry
    else:
        backend = "torch"
        policy_module = torch.load(path)
        def policy(inp):
            inp = TensorDict(tree_map(torch.as_tensor, inp), []).unsqueeze(0)
            out = policy_module(inp)
            action = out["action"].numpy().reshape(-1)
            carry = dict(out["next"].squeeze(0))
            return action, carry

    robot._robot.set_command(init_pos)
    obs = robot.reset()
    obs = robot.get_obs()
    print(obs.shape)

    try:
        inp = {
            "command": robot.command[None, ...],
            "policy": obs[None, ...],
            "is_init": np.array([True]),
            "adapt_hx": np.zeros((1, 128), dtype=np.float32),
        }
        inf_time_sum = 0
        for i in itertools.count():
            iter_start = time.perf_counter()

            inp["command"]  = robot.command[None, ...]
            inp["policy"]   = robot.get_obs()[None, ...]
            inp["is_init"]  = np.array([False])
            action, carry = policy(inp)
            inf_time = time.perf_counter() - iter_start
            inf_time_sum += inf_time

            robot.apply_action(action)
            inp = carry

            if i % 25 == 0:
                # print(robot.projected_gravity)
                print(robot.command)
                print(inf_time_sum / (i + 1))
                # print(robot.robot_state.rpy)
                # print(robot.robot_state.acc - robot.acc_bias, robot.robot_state.gyro)
                # print(robot.robot_state.time_since_state_update, robot.robot_state.state_update_interval)
                # print(robot.jpos_sdk.reshape(4, 3))
                # print(robot.sdk_to_orbit(robot.jpos_sdk).reshape(3, 4))

            time.sleep(max(0, 0.02 - (time.perf_counter() - iter_start)))

    except KeyboardInterrupt:
        print("End")
        
if __name__ == "__main__":
    main()

    
