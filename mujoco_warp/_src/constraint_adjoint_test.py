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

"""Tests for the constraint adjoint."""

import numpy as np
import warp as wp
from absl.testing import absltest
from absl.testing import parameterized

import mujoco_warp as mjw
from mujoco_warp import test_data

# a box on a friction plane, with six single-dof joints so that qpos and qvel are plain coordinates
_BOX_XML = """
<mujoco>
  <option timestep="0.002" cone="{cone}" solver="Newton" iterations="100" ls_iterations="50" tolerance="0"/>
  <worldbody>
    <geom type="plane" size="1 1 .1" friction="0.8 0.005 0.0001"/>
    <body>
      <joint type="slide" axis="1 0 0"/>
      <joint type="slide" axis="0 1 0"/>
      <joint type="slide" axis="0 0 1"/>
      <joint type="hinge" axis="1 0 0"/>
      <joint type="hinge" axis="0 1 0"/>
      <joint type="hinge" axis="0 0 1"/>
      <geom type="box" size=".1 .1 .1" mass="1" friction="0.8 0.005 0.0001"/>
    </body>
  </worldbody>
</mujoco>
"""


def _step(m, d, d_out, qpos, qvel):
  d.qpos.assign(qpos.astype(np.float32))
  d.qvel.assign(qvel.astype(np.float32))
  d.qacc_warmstart.zero_()
  mjw.step(m, d, d_out)
  return d_out.qpos.numpy().astype(np.float64), d_out.qvel.numpy().astype(np.float64)


class ConstraintAdjointTest(parameterized.TestCase):
  @classmethod
  def setUpClass(cls):
    super().setUpClass()
    mjw.enable_grad()

  @parameterized.parameters("pyramidal", "elliptic")
  def test_contact_state_vjp(self, cone):
    """Tests the one-step qpos / qvel VJP of a sliding box against central finite differences."""
    # the box slides along +x and -x, so a pyramid edge is SATISFIED while the others carry force
    qpos = np.array([[0.0, 0.0, 0.0995, 0.01, -0.01, 0.0]] * 2)
    qvel = np.array([[0.5, 0.0, 0.0, 0.0, 0.0, 0.0], [-0.5, 0.0, 0.0, 0.0, 0.0, 0.0]])
    nworld, nv = qvel.shape
    mjm, mjd, m, d = test_data.fixture(xml=_BOX_XML.format(cone=cone), nworld=nworld, nconmax=8, njmax=64)
    d_out = mjw.put_data(mjm, mjd, nworld=nworld, nconmax=8, njmax=64)

    rng = np.random.default_rng(0)
    gq = rng.normal(size=(nworld, nv))
    gv = rng.normal(size=(nworld, nv))

    d.qpos.requires_grad = True
    d.qvel.requires_grad = True
    d.qpos.assign(qpos.astype(np.float32))
    d.qvel.assign(qvel.astype(np.float32))
    d.qacc_warmstart.zero_()
    tape = wp.Tape()
    with tape:
      mjw.step(m, d, d_out)
    d_out.qpos.grad.assign(gq.astype(np.float32))
    d_out.qvel.grad.assign(gv.astype(np.float32))
    tape.backward()
    adj = {"qpos": d.qpos.grad.numpy().astype(np.float64), "qvel": d.qvel.grad.numpy().astype(np.float64)}

    # central finite differences of L = <gq, qpos'> + <gv, qvel'>
    eps = {"qpos": 3e-5, "qvel": 3e-4}
    fd = {"qpos": np.zeros((nworld, nv)), "qvel": np.zeros((nworld, nv))}
    for name, x in (("qpos", qpos), ("qvel", qvel)):
      for i in range(nv):
        loss = []
        for sign in (1.0, -1.0):
          xp = x.copy()
          xp[:, i] += sign * eps[name]
          q1, v1 = _step(m, d, d_out, xp if name == "qpos" else qpos, xp if name == "qvel" else qvel)
          loss.append((gq * q1).sum(axis=1) + (gv * v1).sum(axis=1))
        fd[name][:, i] = (loss[0] - loss[1]) / (2 * eps[name])

    for name in ("qpos", "qvel"):
      for w in range(nworld):
        a, b = adj[name][w], fd[name][w]
        cos = a @ b / (np.linalg.norm(a) * np.linalg.norm(b))
        rel = np.linalg.norm(a - b) / np.linalg.norm(b)
        self.assertGreater(cos, 0.999, f"{name} cosine, world {w}")
        self.assertLess(rel, 0.05, f"{name} relative error, world {w}")


if __name__ == "__main__":
  wp.clear_kernel_cache()
  absltest.main()
