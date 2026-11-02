"""A Gymnasium environment for velocity-tracking locomotion on the Unitree G1,
simulated in Drake with the CENIC error-controlled integrator."""

import gymnasium as gym
import numpy as np
from gymnasium import spaces
from pydrake.geometry import (
    AddContactMaterial,
    AddRigidHydroelasticProperties,
    HalfSpace,
    ProximityProperties,
    StartMeshcat,
)
from pydrake.math import RigidTransform, RollPitchYaw
from pydrake.multibody.plant import (
    AddMultibodyPlantSceneGraph,
    CoulombFriction,
)
from pydrake.systems.analysis import (
    ApplySimulatorConfig,
    Simulator,
    SimulatorConfig,
)
from pydrake.systems.framework import DiagramBuilder
from pydrake.visualization import AddDefaultVisualization

from g1_walking.parsing import make_parser

# mjlab's KNEES_BENT_KEYFRAME for the G1. Unlisted joints are zero.
HOME_JOINTS = {
    "left_hip_pitch_joint": -0.312,
    "right_hip_pitch_joint": -0.312,
    "left_knee_joint": 0.669,
    "right_knee_joint": 0.669,
    "left_ankle_pitch_joint": -0.363,
    "right_ankle_pitch_joint": -0.363,
    "left_shoulder_pitch_joint": 0.2,
    "right_shoulder_pitch_joint": 0.2,
    "left_shoulder_roll_joint": 0.2,
    "right_shoulder_roll_joint": -0.2,
    "left_elbow_joint": 0.6,
    "right_elbow_joint": 0.6,
}
HOME_PELVIS_HEIGHT = 0.76  # m, feet just touching the ground at HOME_JOINTS


class G1WalkEnv(gym.Env):
    """Track a commanded planar velocity with the G1 humanoid.

    Each episode samples a command (forward speed, lateral speed, yaw rate)
    in the pelvis frame. The policy runs at `control_rate` Hz and outputs
    joint position targets for the robot's built-in PD controllers.

    Observations (99,):
        - Pelvis linear velocity in the pelvis frame (3)
        - Pelvis angular velocity in the pelvis frame (3)
        - Gravity direction in the pelvis frame (3)
        - Velocity command [vx, vy, yaw_rate] (3)
        - Joint positions relative to the home pose (29)
        - Joint velocities (29)
        - The previous action (29)

    Actions (29,), nominally in [-1, 1]:
        - Joint position targets as offsets from the home pose, scaled by
          `action_scale` radians. The PD gains and effort limits are those
          in g1.sdf.

    Termination:
        - The pelvis drops below `min_pelvis_height`, or tilts more than
          `max_tilt` radians from upright, or the simulation fails.
    """

    metadata = {"render_modes": ["human"], "render_fps": 50}

    control_rate = 50.0  # Hz
    action_scale = 0.25  # rad per unit action
    min_pelvis_height = 0.45  # m
    max_tilt = 1.0  # rad
    command_ranges = {
        "vx": (-0.5, 1.0),  # m/s
        "vy": (-0.3, 0.3),  # m/s
        "yaw_rate": (-0.5, 0.5),  # rad/s
    }
    reward_weights = {
        "track_lin_vel": 1.0,
        "track_ang_vel": 0.5,
        "lin_vel_z": -2.0,
        "ang_vel_xy": -0.05,
        "orientation": -1.0,
        "action_rate": -0.01,
        "joint_vel": -1e-4,
        "alive": 0.5,
    }

    def __init__(
        self,
        render_mode: str | None = None,
        accuracy: float = 1e-3,
        command: np.ndarray | None = None,
    ):
        """
        Args:
            render_mode: "human" opens meshcat and runs in ~realtime;
                anything else runs headless as fast as possible.
            accuracy: Error-control tolerance for CENIC.
            command: A fixed [vx, vy, yaw_rate] command. If None, a random
                command is sampled at every reset.
        """
        self.render_mode = render_mode
        self.fixed_command = command
        self.meshcat = StartMeshcat() if render_mode == "human" else None

        self._build_diagram()
        self.simulator_config = SimulatorConfig(
            integration_scheme="cenic",
            accuracy=accuracy,
            use_error_control=True,
            target_realtime_rate=1.0 if render_mode == "human" else 0.0,
        )
        self.simulator = None  # Created in reset().

        n = self.num_joints
        self.action_space = spaces.Box(-1.0, 1.0, (n,), np.float32)
        self.observation_space = spaces.Box(
            -np.inf, np.inf, (12 + 3 * n,), np.float32
        )
        self.command = np.zeros(3)
        self.last_action = np.zeros(n)

    def _build_diagram(self):
        builder = DiagramBuilder()
        # CENIC requires a continuous-time plant.
        self.plant, _ = AddMultibodyPlantSceneGraph(builder, time_step=0.0)
        (self.model,) = make_parser(self.plant).AddModelsFromUrl(
            "package://g1_walking/models/unitree_g1/g1.sdf"
        )
        # The G1's collision geometry is compliant hydroelastic, so the
        # ground is rigid hydroelastic.
        ground = ProximityProperties()
        AddRigidHydroelasticProperties(properties=ground)
        AddContactMaterial(
            friction=CoulombFriction(1.0, 1.0), properties=ground
        )
        self.plant.RegisterCollisionGeometry(
            self.plant.world_body(),
            RigidTransform(),
            HalfSpace(),
            "ground",
            ground,
        )
        self.plant.Finalize()

        if self.meshcat is not None:
            AddDefaultVisualization(builder, self.meshcat)

        self.diagram = builder.Build()
        self.context = self.diagram.CreateDefaultContext()
        self.plant_context = self.plant.GetMyMutableContextFromRoot(
            self.context
        )

        # Actions, observations and the PD desired state all use actuator
        # order.
        actuators = [
            self.plant.get_joint_actuator(i)
            for i in self.plant.GetJointActuatorIndices(self.model)
        ]
        self.joints = [a.joint() for a in actuators]
        self.num_joints = len(self.joints)
        self.joint_names = [j.name() for j in self.joints]
        self.q_index = np.array([j.position_start() for j in self.joints])
        self.v_index = np.array([j.velocity_start() for j in self.joints])
        self.home = np.array(
            [HOME_JOINTS.get(name, 0.0) for name in self.joint_names]
        )
        self.pelvis = self.plant.GetBodyByName("pelvis")
        self.desired_state_port = self.plant.get_desired_state_input_port(
            self.model
        )

    def _set_targets(self, action):
        q_desired = self.home + self.action_scale * action
        self.desired_state_port.FixValue(
            self.plant_context,
            np.concatenate([q_desired, np.zeros(self.num_joints)]),
        )

    def _sample_command(self):
        if self.fixed_command is not None:
            return np.asarray(self.fixed_command, dtype=float)
        return np.array(
            [self.np_random.uniform(*r) for r in self.command_ranges.values()]
        )

    def _pelvis_state(self):
        """Pelvis linear/angular velocity and gravity, in the pelvis frame."""
        X_WB = self.pelvis.EvalPoseInWorld(self.plant_context)
        V_WB = self.pelvis.EvalSpatialVelocityInWorld(self.plant_context)
        R_BW = X_WB.rotation().inverse()
        return (
            X_WB,
            R_BW @ V_WB.translational(),
            R_BW @ V_WB.rotational(),
            R_BW @ np.array([0.0, 0.0, -1.0]),
        )

    def _get_obs(self):
        _, lin_vel, ang_vel, gravity = self._pelvis_state()
        q = self.plant.GetPositions(self.plant_context)
        v = self.plant.GetVelocities(self.plant_context)
        return np.concatenate(
            [
                lin_vel,
                ang_vel,
                gravity,
                self.command,
                q[self.q_index] - self.home,
                v[self.v_index],
                self.last_action,
            ]
        ).astype(np.float32)

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        self.context.SetTime(0.0)
        self.diagram.SetDefaultContext(self.context)

        q_joints = self.home + self.np_random.uniform(
            -0.05, 0.05, self.num_joints
        )
        for joint, q in zip(self.joints, q_joints, strict=True):
            joint.set_angle(self.plant_context, q)
        yaw = self.np_random.uniform(-np.pi, np.pi)
        self.plant.SetFreeBodyPose(
            self.plant_context,
            self.pelvis,
            RigidTransform(
                RollPitchYaw(0.0, 0.0, yaw), [0.0, 0.0, HOME_PELVIS_HEIGHT]
            ),
        )

        self.command = self._sample_command()
        self.last_action = np.zeros(self.num_joints)
        self._set_targets(self.last_action)
        # A fresh simulator per episode, because CENIC carries step size
        # and solver warm starts between steps, which would make episodes
        # depend on the previous one.
        self.simulator = Simulator(self.diagram, self.context)
        ApplySimulatorConfig(self.simulator_config, self.simulator)
        self.simulator.Initialize()
        return self._get_obs(), {"command": self.command.copy()}

    def step(self, action):
        action = np.clip(np.asarray(action, dtype=float), -1.0, 1.0)
        self._set_targets(action)

        dt = 1.0 / self.control_rate
        try:
            self.simulator.AdvanceTo(self.context.get_time() + dt)
            sim_failed = False
        except RuntimeError:
            sim_failed = True

        X_WB, lin_vel, ang_vel, gravity = self._pelvis_state()
        v = self.plant.GetVelocities(self.plant_context)
        tilt = np.arccos(np.clip(-gravity[2], -1.0, 1.0))
        terminated = bool(
            sim_failed
            or X_WB.translation()[2] < self.min_pelvis_height
            or tilt > self.max_tilt
        )

        terms = {
            "track_lin_vel": np.exp(
                -np.sum((self.command[:2] - lin_vel[:2]) ** 2) / 0.25
            ),
            "track_ang_vel": np.exp(
                -((self.command[2] - ang_vel[2]) ** 2) / 0.25
            ),
            "lin_vel_z": lin_vel[2] ** 2,
            "ang_vel_xy": np.sum(ang_vel[:2] ** 2),
            "orientation": np.sum(gravity[:2] ** 2),
            "action_rate": np.sum((action - self.last_action) ** 2),
            "joint_vel": np.sum(v[self.v_index] ** 2),
            "alive": 0.0 if terminated else 1.0,
        }
        reward = dt * sum(self.reward_weights[k] * terms[k] for k in terms)

        self.last_action = action
        obs = self._get_obs()
        if not np.all(np.isfinite(obs)):
            obs = np.nan_to_num(obs, nan=0.0, posinf=0.0, neginf=0.0)
            terminated = True
        info = {"reward_terms": terms, "sim_failed": sim_failed}
        return obs, float(reward), terminated, False, info

    def render(self):
        # Meshcat updates itself during simulation when render_mode="human".
        pass
