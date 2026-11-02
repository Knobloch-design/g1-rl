# Unitree G1

A 29-DOF Unitree G1 humanoid with a floating pelvis, for Drake.

## Source

`g1.sdf` and `meshes/` are converted from the G1 model in
[mjlab](https://github.com/mujocolab/mjlab) at commit
[`a6a76e59bf`](https://github.com/mujocolab/mjlab/tree/a6a76e59bf021fa79c5bb3e13e0279456bf5cfa7/src/mjlab/asset_zoo/robots/unitree_g1):
`xmls/g1.xml` for the model and `g1_constants.py` for the actuator parameters.
The source STL meshes are identical to those in
[MuJoCo Menagerie's unitree_g1](https://github.com/google-deepmind/mujoco_menagerie/tree/main/unitree_g1),
which is derived from Unitree's
[g1_description](https://github.com/unitreerobotics/unitree_ros/tree/master/robots/g1_description)
(`g1_29dof_rev_1_0`). See [LICENSE.md](LICENSE.md) for the licenses.

## Conversion

1. The STL meshes were converted to OBJ with trimesh, because Drake can't load
   STL.
2. `g1.xml` was loaded with Drake's MJCF parser, which resolves default
   classes, capsule `fromto` and quaternions.
3. The SDF was written from the parsed plant: links with poses and inertias,
   revolute joints with axes and position limits, visual meshes with mjlab's
   colors, collision capsules and spheres, and frames for mjlab's sites (IMUs,
   feet, palms). The pelvis has no joint, so Drake gives it a floating base.

The conversion script is not part of this repository.

## Added

- Effort limits and armature from mjlab's `g1_constants.py`. Armature is
  written as `drake:rotor_inertia` with a gear ratio of 1. The waist pitch/roll
  and ankle limits are mjlab's approximation of a linkage driven by two motors,
  not a Unitree spec.
- PD gains from mjlab's `g1_constants.py`, written as `drake:controller_gains`
  on every joint. mjlab sets the stiffness from the armature and a 10 Hz
  natural frequency, and the damping for a damping ratio of 2. A task can
  override them with `JointActuator.set_controller_gains()` before
  `Finalize()`. The PD is only active when the task connects the plant's
  desired state input port, and the effort limit clamps the PD plus the
  actuation input. On their own, these gains don't hold the robot standing at
  mjlab's HOME pose.
- Hydroelastic contact properties and friction on every collision geometry.
- Collision filter groups for mjlab's `<contact><exclude>` pairs.

The values are in `g1.sdf`.

## Dropped

- MuJoCo sensors, the light and the camera. Drake doesn't use them, and sensors
  belong in tasks.
- Velocity limits. mjlab defines them but doesn't apply them.

## Contact

The collision geometry is mjlab's set of capsules and spheres, which only
roughly follows the visual meshes. The meshes are visual only. Every collision
geometry is compliant hydroelastic, so the ground and other objects in a task
should be rigid hydroelastic. A `HalfSpace` works well for the ground.

The hydroelastic modulus and dissipation are not from mjlab, because MuJoCo's
contact settings don't map onto them. They were chosen by simulating the robot
standing, landing and falling on rigid ground. Friction is set on every
collision geometry, not only the feet as in mjlab, so that the arms and hands
can be used in contact-rich tasks.

The hydroelastic meshes are coarse: each `drake:mesh_resolution_hint` equals
the radius of its capsule or sphere. That is acceptable here because the
collision shapes are approximate anyway and contact at the feet matters most.
Use a smaller hint if a task needs more accurate contact elsewhere.
