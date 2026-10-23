# Copyright 2026 The Newton Developers
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

"""Tests for the forward adjoint."""

import numpy as np
import warp as wp
from absl.testing import absltest

import mujoco_warp as mjw
from mujoco_warp import test_data

# a mass on a slide joint, driven by a position actuator whose force is clamped by actuatorfrcrange
_XML = """
<mujoco>
  <option timestep="0.01" gravity="0 0 0"/>
  <worldbody>
    <body>
      <joint name="j" type="slide" axis="1 0 0" actuatorfrclimited="true" actuatorfrcrange="-0.5 0.5"/>
      <geom type="sphere" size=".05" mass="1"/>
    </body>
  </worldbody>
  <actuator>
    <position joint="j" kp="10"/>
  </actuator>
</mujoco>
"""


class ForwardAdjointTest(absltest.TestCase):
  @classmethod
  def setUpClass(cls):
    super().setUpClass()
    mjw.enable_grad()

  def test_actuator_force_clamp_vjp(self):
    """Tests the one-step VJP with a clamped joint actuator force against finite differences."""
    # kp * (ctrl - qpos) = 0.2, 5, -4 against the force range +-0.5: the last two worlds are clamped
    qpos = np.zeros((3, 1))
    qvel = np.array([[0.1], [-0.2], [0.3]])
    ctrl = np.array([[0.02], [0.5], [-0.4]])
    nworld = 3
    mjm, mjd, m, d = test_data.fixture(xml=_XML, nworld=nworld)
    d_out = mjw.put_data(mjm, mjd, nworld=nworld)

    def step(q, v, u):
      d.qpos.assign(q.astype(np.float32))
      d.qvel.assign(v.astype(np.float32))
      d.ctrl.assign(u.astype(np.float32))
      mjw.step(m, d, d_out)
      return d_out.qpos.numpy().astype(np.float64) + d_out.qvel.numpy().astype(np.float64)

    # adjoint of L = qpos' + qvel'
    for array in (d.qpos, d.qvel, d.ctrl):
      array.requires_grad = True
    d.qpos.assign(qpos.astype(np.float32))
    d.qvel.assign(qvel.astype(np.float32))
    d.ctrl.assign(ctrl.astype(np.float32))
    tape = wp.Tape()
    with tape:
      mjw.step(m, d, d_out)
    d_out.qpos.grad.fill_(1.0)
    d_out.qvel.grad.fill_(1.0)
    tape.backward()

    eps = 1e-3
    for name, x, index in (("qpos", qpos, 0), ("qvel", qvel, 1), ("ctrl", ctrl, 2)):
      args = [qpos, qvel, ctrl]
      args_p = list(args)
      args_m = list(args)
      args_p[index] = x + eps
      args_m[index] = x - eps
      fd = (step(*args_p) - step(*args_m)) / (2 * eps)
      adj = getattr(d, name).grad.numpy().astype(np.float64)
      np.testing.assert_allclose(adj, fd, atol=1e-3, err_msg=f"mismatch: {name}")


if __name__ == "__main__":
  wp.clear_kernel_cache()
  absltest.main()
