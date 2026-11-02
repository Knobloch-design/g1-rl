import gymnasium as gym
import numpy as np
import pytest
from gymnasium.utils.env_checker import check_env

import g1_walking  # noqa: F401  (registers G1Walk-v0)
from g1_walking import G1WalkEnv


@pytest.fixture(scope="module")
def env():
    """Share one headless env across this module's tests."""
    return G1WalkEnv()


def test_spaces(env):
    assert env.action_space.shape == (29,)
    assert env.observation_space.shape == (99,)
    obs, info = env.reset(seed=0)
    assert obs.shape == (99,)
    assert info["command"].shape == (3,)


def test_reset_is_deterministic(env):
    obs_a, _ = env.reset(seed=3)
    obs_b, _ = env.reset(seed=3)
    obs_c, _ = env.reset(seed=4)
    np.testing.assert_array_equal(obs_a, obs_b)
    assert not np.array_equal(obs_a, obs_c)


def test_home_pose_starts_upright_and_still(env):
    obs, _ = env.reset(seed=0)
    np.testing.assert_allclose(obs[6:9], [0.0, 0.0, -1.0], atol=1e-9)
    np.testing.assert_allclose(obs[:6], 0.0, atol=1e-9)
    np.testing.assert_allclose(obs[12:41], 0.0, atol=0.05)


def test_zero_action_survives_briefly(env):
    """Holding the home pose with PD shouldn't fall within 0.5 s."""
    env.reset(seed=0)
    for _ in range(25):
        obs, reward, terminated, truncated, info = env.step(np.zeros(29))
        assert not terminated
        assert np.isfinite(reward)
    assert not info["sim_failed"]


def test_falling_terminates(env):
    """Pushing every joint to an extreme topples the robot."""
    env.reset(seed=0)
    for _ in range(250):
        _, _, terminated, _, _ = env.step(np.ones(29))
        if terminated:
            return
    pytest.fail("robot never fell")


def test_fixed_command():
    env = G1WalkEnv(command=[0.5, 0.0, 0.0])
    _, info = env.reset(seed=1)
    np.testing.assert_array_equal(info["command"], [0.5, 0.0, 0.0])


def test_gymnasium_api(env):
    check_env(env, skip_render_check=True)


def test_registered():
    env = gym.make("G1Walk-v0")
    assert env.spec.max_episode_steps == 1000


def test_step_is_deterministic_across_episodes(env):
    """Episodes don't leak integrator state into the next one."""
    action = np.full(29, 0.3)
    env.reset(seed=5)
    obs_a = env.step(action)[0]
    env.step(action)
    env.reset(seed=5)
    obs_b = env.step(action)[0]
    np.testing.assert_array_equal(obs_a, obs_b)
