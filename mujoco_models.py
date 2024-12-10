#reference: https://mujoco.readthedocs.io/en/stable/overview.html#examples
XML_BOX = """
<mujoco>
  <worldbody>
    <light diffuse=".5 .5 .5" pos="0 0 3" dir="0 0 -1"/>
    <geom type="plane" size="1 1 0.1" rgba=".9 0 0 1"/>
    <body pos="0 0 1">
      <joint type="free"/>
      <geom type="box" size=".1 .2 .3" rgba="0 .9 0 1"/>
    </body>
  </worldbody>
</mujoco>
"""

# reference: https://colab.research.google.com/github/google-deepmind/mujoco/blob/main/mjx/tutorial.ipynb#scrollTo=bNus3mbbDz6a

XML_BOX_AND_BALL = """
<mujoco>
  <worldbody>
    <light name="top" pos="0 0 1"/>
    <body name="box_and_sphere" euler="0 0 -30">
      <joint name="swing" type="hinge" axis="1 -1 0" pos="-.2 -.2 -.2"/>
      <geom name="red_box" type="box" size=".2 .2 .2" rgba="1 0 0 1"/>
      <geom name="green_sphere" pos=".2 .2 .2" size=".1" rgba="0 1 0 1"/>
    </body>
  </worldbody>
</mujoco>
"""

#reference: https://mujoco.readthedocs.io/en/stable/overview.html#examples
XML_ARM_WITH_ROPE = """
<mujoco model="example">
  <default>
    <geom rgba=".8 .6 .4 1"/>
  </default>

  <asset>
    <texture type="skybox" builtin="gradient" rgb1="1 1 1" rgb2=".6 .8 1" width="256" height="256"/>
  </asset>

  <worldbody>
    <light pos="0 1 1" dir="0 -1 -1" diffuse="1 1 1"/>
    <body pos="0 0 1">
      <joint type="ball"/>
      <geom type="capsule" size="0.06" fromto="0 0 0  0 0 -.4"/>
      <body pos="0 0 -0.4">
        <joint axis="0 1 0"/>
        <joint axis="1 0 0"/>
        <geom type="capsule" size="0.04" fromto="0 0 0  .3 0 0"/>
        <body pos=".3 0 0">
          <joint axis="0 1 0"/>
          <joint axis="0 0 1"/>
          <geom pos=".1 0 0" size="0.1 0.08 0.02" type="ellipsoid"/>
          <site name="end1" pos="0.2 0 0" size="0.01"/>
        </body>
      </body>
    </body>

    <body pos="0.3 0 0.1">
      <joint type="free"/>
      <geom size="0.07 0.1" type="cylinder"/>
      <site name="end2" pos="0 0 0.1" size="0.01"/>
    </body>
  </worldbody>

  <tendon>
    <spatial limited="true" range="0 0.6" width="0.005">
      <site site="end1"/>
      <site site="end2"/>
    </spatial>
  </tendon>
</mujoco>
    """


# reference: https://colab.research.google.com/github/google-deepmind/mujoco/blob/main/mjx/tutorial.ipynb#scrollTo=mtGMYNLE3QJN
# path /usr/local/lib/python3.10/dist-packages/mujoco/mjx/test_data/humanoid

XML_HUMANOID = """
<mujoco model="01 Humanoids">
  <option timestep="0.005" solver="Newton" iterations="1" ls_iterations="4">
    <flag eulerdamp="disable"/>
  </option>

  <custom>
    <numeric data="4" name="max_contact_points"/>
  </custom>

  <size memory="100M"/>

  <asset>
    <texture type="skybox" builtin="gradient" rgb1=".3 .5 .7" rgb2="0 0 0" width="512" height="512"/>
    <texture name="body" type="cube" builtin="flat" mark="cross" width="128" height="128"
             rgb1="0.8 0.6 0.4" rgb2="0.8 0.6 0.4" markrgb="1 1 1" random="0.01"/>
    <material name="body" texture="body" texuniform="true" rgba="0.8 0.6 .4 1"/>
    <texture name="grid" type="2d" builtin="checker" width="512" height="512" rgb1=".1 .2 .3" rgb2=".2 .3 .4"/>
    <material name="grid" texture="grid" texrepeat="1 1" texuniform="true" reflectance=".2"/>
  </asset>

  <default>
    <motor ctrlrange="-1 1" ctrllimited="true"/>
    <default class="body">
      <geom type="capsule" condim="3" friction=".7" solimp=".9 .99 .003" solref=".015 1" material="body" contype="0"/>
      <joint type="hinge" damping=".2" stiffness="1" armature=".01" limited="true" solimplimit="0 .99 .01"/>
      <default class="big_joint">
        <joint damping="5" stiffness="10"/>
        <default class="big_stiff_joint">
          <joint stiffness="20"/>
        </default>
      </default>
    </default>
  </default>

  <visual>
    <map force="0.1" zfar="30"/>
    <rgba haze="0.15 0.25 0.35 1"/>
    <quality shadowsize="4096"/>
    <global offwidth="800" offheight="800"/>
  </visual>

  <worldbody>
    <geom size="10 10 .05" type="plane" material="grid" condim="3"/>
    <light dir=".2 1 -.4" diffuse=".8 .8 .8" specular="0.3 0.3 0.3" pos="-2 -10 4" cutoff="35"/>
    <light dir="-.2 1 -.4" diffuse=".8 .8 .8" specular="0.3 0.3 0.3" pos="2 -10 4" cutoff="35"/>

    <body name="1a_torso" pos="-1 0 1.5" childclass="body">
      <camera name="1a_back" pos="-3 0 1" xyaxes="0 -1 0 1 0 2" mode="trackcom"/>
      <camera name="1a_side" pos="0 -3 1" xyaxes="1 0 0 0 1 2" mode="trackcom"/>
      <freejoint name="1a_root"/>
      <geom name="1a_torso" fromto="0 -.07 0 0 .07 0" size=".07"/>
      <geom name="1a_upper_waist" fromto="-.01 -.06 -.12 -.01 .06 -.12" size=".06"/>
      <body name="1a_head" pos="0 0 .19">
        <geom name="1a_head" type="sphere" size=".09"/>
        <camera name="1a_egocentric" pos=".09 0 0" xyaxes="0 -1 0 .1 0 1" fovy="80"/>
      </body>
      <body name="1a_lower_waist" pos="-.01 0 -.26">
        <geom name="1a_lower_waist" fromto="0 -.06 0 0 .06 0" size=".06"/>
        <joint name="1a_abdomen_z" pos="0 0 .065" axis="0 0 1" range="-45 45" class="big_stiff_joint"/>
        <joint name="1a_abdomen_y" pos="0 0 .065" axis="0 1 0" range="-75 30" class="big_joint"/>
        <body name="1a_pelvis" pos="0 0 -.165">
          <joint name="1a_abdomen_x" pos="0 0 .1" axis="1 0 0" range="-35 35" class="big_joint"/>
          <geom name="1a_butt" fromto="-.02 -.07 0 -.02 .07 0" size=".09"/>
          <body name="1a_right_thigh" pos="0 -.1 -.04">
            <joint name="1a_right_hip_x" axis="1 0 0" range="-25 5" class="big_joint"/>
            <joint name="1a_right_hip_z" axis="0 0 1" range="-60 35" class="big_joint"/>
            <joint name="1a_right_hip_y" axis="0 1 0" range="-110 20" class="big_stiff_joint"/>
            <geom name="1a_right_thigh" fromto="0 0 0 0 .01 -.34" size=".06"/>
            <body name="1a_right_shin" pos="0 .01 -.403">
              <joint name="1a_right_knee" pos="0 0 .02" axis="0 -1 0" range="-160 2"/>
              <geom name="1a_right_shin" fromto="0 0 0 0 0 -.3"  size=".049"/>
              <body name="1a_right_foot" pos="0 0 -.39">
                <joint name="1a_right_ankle_y" pos="0 0 .08" axis="0 1 0" range="-50 50" stiffness="6"/>
                <joint name="1a_right_ankle_x" pos="0 0 .04" axis="1 0 .5" range="-50 50" stiffness="3"/>
                <geom name="1a_right_right_foot" fromto="-.07 -.02 0 .14 -.04 0" size=".027"/>
                <geom name="1a_left_right_foot" fromto="-.07 0 0 .14  .02 0" size=".027"/>
              </body>
            </body>
          </body>
          <body name="1a_left_thigh" pos="0 .1 -.04">
            <joint name="1a_left_hip_x" axis="-1 0 0" range="-25 5" class="big_joint"/>
            <joint name="1a_left_hip_z" axis="0 0 -1" range="-60 35" class="big_joint"/>
            <joint name="1a_left_hip_y" axis="0 1 0" range="-110 20" class="big_stiff_joint"/>
            <geom name="1a_left_thigh" fromto="0 0 0 0 -.01 -.34" size=".06"/>
            <body name="1a_left_shin" pos="0 -.01 -.403">
              <joint name="1a_left_knee" pos="0 0 .02" axis="0 -1 0" range="-160 2"/>
              <geom name="1a_left_shin" fromto="0 0 0 0 0 -.3"  size=".049"/>
              <body name="1a_left_foot" pos="0 0 -.39">
                <joint name="1a_left_ankle_y" pos="0 0 .08" axis="0 1 0" range="-50 50" stiffness="6"/>
                <joint name="1a_left_ankle_x" pos="0 0 .04" axis="1 0 .5" range="-50 50" stiffness="3"/>
                <geom name="1a_left_left_foot" fromto="-.07 .02 0 .14 .04 0" size=".027"/>
                <geom name="1a_right_left_foot" fromto="-.07 0 0 .14 -.02 0" size=".027"/>
              </body>
            </body>
          </body>
        </body>
      </body>
      <body name="1a_right_upper_arm" pos="0 -.17 .06">
        <joint name="1a_right_shoulder1" axis="2 1 1"  range="-85 60"/>
        <joint name="1a_right_shoulder2" axis="0 -1 1" range="-85 60"/>
        <geom name="1a_right_upper_arm" fromto="0 0 0 .16 -.16 -.16" size=".04 .16"/>
        <body name="1a_right_lower_arm" pos=".18 -.18 -.18">
          <joint name="1a_right_elbow" axis="0 -1 1" range="-90 50" stiffness="0"/>
          <geom name="1a_right_lower_arm" fromto=".01 .01 .01 .17 .17 .17" size=".031"/>
          <body name="1a_right_hand" pos=".18 .18 .18">
            <geom name="1a_right_hand" type="sphere" size=".04" zaxis="1 1 1"/>
          </body>
        </body>
      </body>
      <body name="1a_left_upper_arm" pos="0 .17 .06">
        <joint name="1a_left_shoulder1" axis="2 -1 1" range="-60 85"/>
        <joint name="1a_left_shoulder2" axis="0 1 1"  range="-60 85"/>
        <geom name="1a_left_upper_arm" fromto="0 0 0 .16 .16 -.16" size=".04 .16"/>
        <body name="1a_left_lower_arm" pos=".18 .18 -.18">
          <joint name="1a_left_elbow" axis="0 -1 -1" range="-90 50" stiffness="0"/>
          <geom name="1a_left_lower_arm" fromto=".01 -.01 .01 .17 -.17 .17" size=".031"/>
          <body name="1a_left_hand" pos=".18 -.18 .18">
            <geom name="1a_left_hand" type="sphere" size=".04" zaxis="1 -1 1"/>
          </body>
        </body>
      </body>
    </body>

  </worldbody>

  <actuator>
    <motor gear="40"  joint="1a_abdomen_y"/>
    <motor gear="40"  joint="1a_abdomen_z"/>
    <motor gear="40"  joint="1a_abdomen_x"/>
    <motor gear="40"  joint="1a_right_hip_x"/>
    <motor gear="40"  joint="1a_right_hip_z"/>
    <motor gear="120" joint="1a_right_hip_y"/>
    <motor gear="80"  joint="1a_right_knee"/>
    <motor gear="20"  joint="1a_right_ankle_x"/>
    <motor gear="20"  joint="1a_right_ankle_y"/>
    <motor gear="40"  joint="1a_left_hip_x"/>
    <motor gear="40"  joint="1a_left_hip_z"/>
    <motor gear="120" joint="1a_left_hip_y"/>
    <motor gear="80"  joint="1a_left_knee"/>
    <motor gear="20"  joint="1a_left_ankle_x"/>
    <motor gear="20"  joint="1a_left_ankle_y"/>
    <motor gear="20"  joint="1a_right_shoulder1"/>
    <motor gear="20"  joint="1a_right_shoulder2"/>
    <motor gear="40"  joint="1a_right_elbow"/>
    <motor gear="20"  joint="1a_left_shoulder1"/>
    <motor gear="20"  joint="1a_left_shoulder2"/>
    <motor gear="40"  joint="1a_left_elbow"/>
  </actuator>

</mujoco>
"""

# reference: https://github.com/google-deepmind/mujoco_menagerie/blob/main/google_barkour_vb/barkour_vb.xml

XML_QUADROPED_VB = """
<mujoco model="barkour_vb">
  <compiler angle="radian" autolimits="true"/>

  <default>
    <default class="bkvb">
      <geom type="mesh"/>
      <!-- hide sites by default -->
      <site group="5"/>
      <default class="bkvb/vicon">
        <!-- show vicon markers with a small sphere -->
        <site group="0"/>
      </default>

      <joint damping="0.024" frictionloss="0.13" armature="0.011"/>
      <general forcerange="-18 18" biastype="affine" gainprm="50 0 0" biasprm="0 -50 -0.5"/>
      <default class="bkvb/abduction">
        <joint range="-1.0472 1.0472"/>
        <general ctrlrange="-0.9472 0.9472"/>
        <geom rgba="0.980392 0.713726 0.00392157 1" mesh="abduction"/>
      </default>
      <default class="bkvb/hip">
        <joint range="-1.54706 3.02902"/>
        <general ctrlrange="-1.44706 2.92902"/>
      </default>
      <default class="bkvb/knee">
        <joint range="0 2.44346"/>
        <general ctrlrange="0.1 2.34346"/>
      </default>
      <default class="bkvb/foot">
        <site pos="-0.21425 -0.0779806 0" quat="0.664463 0.664463 -0.241845 -0.241845"/>
        <geom rgba="0.231373 0.380392 0.705882 1" mesh="foot" solimp="0.015 1 0.031" friction="0.8 0.02 0.01"/>
      </default>
      <default class="bkvb/lower_leg">
        <geom rgba="0.615686 0.811765 0.929412 1" mesh="lower_leg"/>
      </default>
      <default class="bkvb/upper_leg">
        <geom rgba="0.615686 0.811765 0.929412 1" mesh="upper_leg"/>
      </default>
      <default class="bkvb/upper_leg_left">
        <geom rgba="0.972549 0.529412 0.00392157 1" mesh="upper_leg_left"/>
      </default>
      <default class="bkvb/upper_leg_right">
        <geom rgba="0.513726 0.737255 0.407843 1" mesh="upper_leg_right"/>
      </default>
      <default class="bkvb/torso">
        <geom rgba="0.8 0.74902 0.913725 1"/>
      </default>
    </default>
  </default>

  <asset>
    <mesh name="camera_cover" file="assets/camera_cover.stl"/>
    <mesh name="neck" file="assets/neck.stl"/>
    <mesh name="intel_realsense_depth_camera_d435" file="assets/intel_realsense_depth_camera_d435.stl"/>
    <mesh name="handle" file="assets/handle.stl"/>
    <mesh name="torso" file="assets/torso.stl"/>
    <mesh name="abduction" file="assets/abduction.stl"/>
    <mesh name="upper_leg" file="assets/upper_leg.stl"/>
    <mesh name="upper_leg_left" file="assets/upper_leg_left.stl"/>
    <mesh name="upper_leg_right" file="assets/upper_leg_right.stl"/>
    <mesh name="lower_leg" file="assets/lower_leg.stl"/>
    <mesh name="foot" file="assets/foot.stl"/>
  </asset>

  <worldbody>
    <body name="torso" childclass="bkvb">
      <freejoint name="torso"/>
      <camera name="track" pos="0.846 -1.465 0.916" xyaxes="0.866 0.500 0.000 -0.171 0.296 0.940" mode="trackcom"/>
      <inertial pos="0.0055238 -0.000354563 0.00835899" quat="-0.00150849 0.694899 -0.000198355 0.719106" mass="6.04352"
        diaginertia="0.144664 0.12027 0.0511405"/>
      <geom class="bkvb/torso" pos="-7.85127e-05 -0.000500734 0" mesh="neck"/>
      <geom class="bkvb/torso" pos="-7.85127e-05 -0.000500734 0" mesh="camera_cover"/>
      <geom class="bkvb/torso" pos="-7.85127e-05 -0.000500734 0" mesh="handle"/>
      <geom class="bkvb/torso" pos="0.319921 -0.000500734 0.0651248" quat="1 0 0 1"
        mesh="intel_realsense_depth_camera_d435"/>
      <geom class="bkvb/torso" pos="-7.85127e-05 -0.000500734 0" mesh="torso"/>
      <site name="imu_frame" pos="0.010715 -0.00025 -0.06" quat="0 0 0 1"/>
      <site name="base_frame"/>
      <site name="vicon_frame"/>
      <!-- dummy bodies for the cameras -->
      <body pos="0.3176 0.017 0.065" quat="1 -1 1 -1">
        <site name="head_camera_frame"/>
        <camera name="realsense/depth" fovy="62" quat="0 1 0 0"/>
        <site name="realsense/depth_frame" quat="0 1 0 0"/>
        <camera name="realsense/rgb" fovy="42.5" pos="0 0.015 0" quat="0 1 0 0"/>
        <site name="realsense/rgb_frame" pos="0 0.015 0" quat="0 1 0 0"/>
        <site name="realsense/imu"/>
      </body>
      <body pos="0.08632 0 0.1213" quat="1 -1 1 -1" name="oak/">
        <site name="handle_camera_frame"/>
        <camera name="oak/rgb" pos="0 0 0.0125" quat="0 -1 -0 -0" resolution="96 60"
          focalpixel="66.55567095 66.53559105" principalpixel="-1.1446674 -1.8664293" sensorsize="0.000288 0.00018"/>
      </body>
      <site name="vicon_0" class="bkvb/vicon" pos="-0.110864 -0.117663 0.061736"/>
      <site name="vicon_1" class="bkvb/vicon" pos="-0.110864 0.116662 0.0617363"
        quat="0.965087 -0.209027 0.154269 -0.0334129"/>
      <site name="vicon_2" class="bkvb/vicon" pos="-0.0625744 0.11068 0.0458139" quat="0.900548 -0.434757 0 0"/>
      <site name="vicon_3" class="bkvb/vicon" pos="0.0561715 0.100042 0.0527394" quat="0.98926 -0.146166 0 0"/>
      <site name="vicon_4" class="bkvb/vicon" pos="0.0960704 0.129697 0.0389732"
        quat="0.767626 -0.517423 -0.313596 0.211381"/>
      <site name="vicon_5" class="bkvb/vicon" pos="-0.0962327 0.13163 -0.0388594"
        quat="0.529619 -0.759572 0.215948 -0.309709"/>
      <site name="vicon_6" class="bkvb/vicon" pos="-7.85127e-05 -0.0778626 0.070357" quat="0.94293 0.33299 0 0"/>
      <site name="vicon_8" class="bkvb/vicon" pos="-0.0962274 -0.130699 0.0389732"
        quat="0.767626 0.517423 0.313596 0.211381"/>
      <site name="vicon_9" class="bkvb/vicon" pos="0.0561715 -0.101043 0.0527394" quat="0.98926 0.146166 0 0"/>
      <body name="leg_front_left" pos="0.171671 0.0892493 -9.8e-06" quat="1 -1 -1 1">
        <inertial pos="0.00547726 -0.000288034 -0.0602191" quat="0.999837 0.0103892 -0.0143715 -0.00325656" mass="0.787"
          diaginertia="0.00143831 0.00117023 0.00100011"/>
        <joint name="abduction_front_left" class="bkvb/abduction"/>
        <geom class="bkvb/abduction" pos="0 0.000111373 0.0029" quat="1 1 0 0"/>
        <body name="upper_leg_front_left" pos="0.03085 0 -0.065" quat="0 -1 0 1">
          <inertial pos="-0.0241397 0.00402429 -0.0453038" quat="0.0673193 0.647966 -0.00518142 0.75867" mass="1.155"
            diaginertia="0.00562022 0.00519471 0.0012633"/>
          <joint name="hip_front_left" class="bkvb/hip"/>
          <geom class="bkvb/upper_leg" pos="0.0679 0.000111373 0.03085" quat="1 1 1 -1"/>
          <geom class="bkvb/upper_leg_left" pos="0 0 -0.05075" quat="0 0 1 0"/>
          <body name="lower_leg_front_left" pos="-0.19 0 -0.069575" quat="0 0 1 0">
            <inertial pos="-0.0895493 -0.0301957 -3.02082e-08" quat="-0.101465 0.699789 0.101465 0.699789"
              mass="0.171238" diaginertia="0.00137406 0.00135746 3.05521e-05"/>
            <joint name="knee_front_left" class="bkvb/knee"/>
            <geom class="bkvb/foot" pos="-0.0649838 0.178542 0" quat="0.819152 0 0 -0.573576"/>
            <geom class="bkvb/lower_leg" pos="-0.0649838 0.178542 0" quat="0.819152 0 0 -0.573576"/>
            <site name="foot_front_left" class="bkvb/foot"/>
          </body>
        </body>
      </body>
      <body name="leg_hind_left" pos="-0.171829 0.0892493 -9.8e-06" quat="1 -1 -1 1">
        <inertial pos="0.00547726 0.000288034 0.0602191" quat="0.999837 0.0103892 0.0143715 0.00325656" mass="0.787"
          diaginertia="0.00143831 0.00117023 0.00100011"/>
        <joint name="abduction_hind_left" class="bkvb/abduction"/>
        <geom class="bkvb/abduction" pos="0 -0.000111373 -0.0029" quat="1 -1 0 0"/>
        <body name="upper_leg_hind_left" pos="0.03085 0 0.065" quat="0 1 0 -1">
          <inertial pos="-0.0241397 0.00402429 -0.0453038" quat="0.0673193 0.647966 -0.00518142 0.75867" mass="1.155"
            diaginertia="0.00562022 0.00519471 0.0012633"/>
          <joint name="hip_hind_left" class="bkvb/hip"/>
          <geom class="bkvb/upper_leg" pos="0.0679 0.000111373 0.03085" quat="1 1 1 -1"/>
          <geom class="bkvb/upper_leg_left" pos="0 0 -0.05075" quat="0 0 1 0"/>
          <body name="lower_leg_2" pos="-0.19 0 -0.069925" quat="0 0 1 0">
            <inertial pos="-0.0895493 -0.0301957 -3.02082e-08" quat="-0.101465 0.699789 0.101465 0.699789"
              mass="0.171238" diaginertia="0.00137406 0.00135746 3.05521e-05"/>
            <joint name="knee_hind_left" class="bkvb/knee"/>
            <geom class="bkvb/lower_leg" pos="-0.0649838 0.178542 0" quat="0.819152 0 0 -0.573576"/>
            <geom class="bkvb/foot" pos="-0.0649838 0.178542 0" quat="0.819152 0 0 -0.573576"/>
            <site name="foot_hind_left" class="bkvb/foot"/>
          </body>
        </body>
      </body>
      <body name="leg_front_right" pos="0.171671 -0.0907507 -9.8e-06" quat="1 -1 1 -1">
        <inertial pos="0.00547726 0.000288034 0.0602191" quat="0.999837 0.0103892 0.0143715 0.00325656" mass="0.787"
          diaginertia="0.00143831 0.00117023 0.00100011"/>
        <joint name="abduction_front_right" class="bkvb/abduction"/>
        <geom class="bkvb/abduction" pos="0 -0.000111373 -0.0029" quat="1 -1 0 0"/>
        <body name="upper_leg_front_right" pos="0.03085 0 0.065" quat="0 -1 0 -1">
          <inertial pos="-0.0241393 0.00324567 0.0453036" quat="-0.00604983 0.756969 -0.0854547 0.64781" mass="1.155"
            diaginertia="0.00563107 0.00519539 0.00126472"/>
          <joint name="hip_front_right" class="bkvb/hip"/>
          <geom class="bkvb/upper_leg" pos="0.0679 -0.000111373 -0.03085" quat="1 -1 -1 -1"/>
          <geom class="bkvb/upper_leg_right" pos="0 0 0.05075" quat="0 0 -1 0"/>
          <body name="lower_leg_3" pos="-0.19 0 0.069575" quat="0 0 -1 0">
            <inertial pos="-0.0895493 -0.0301957 -3.02082e-08" quat="-0.101465 0.699789 0.101465 0.699789"
              mass="0.171238" diaginertia="0.00137406 0.00135746 3.05521e-05"/>
            <joint name="knee_front_right" class="bkvb/knee"/>
            <geom class="bkvb/foot" pos="-0.0649838 0.178542 0" quat="0.819152 0 0 -0.573576"/>
            <geom class="bkvb/lower_leg" pos="-0.0649838 0.178542 0" quat="0.819152 0 0 -0.573576"/>
            <site name="foot_front_right" class="bkvb/foot"/>
          </body>
        </body>
      </body>
      <body name="leg_hind_right" pos="-0.171829 -0.0907507 -9.8e-06" quat="1 -1 1 -1">
        <inertial pos="0.00547726 -0.000288034 -0.0600191" quat="0.999837 0.0103892 -0.0143715 -0.00325656" mass="0.787"
          diaginertia="0.00143831 0.00117023 0.00100011"/>
        <joint name="abduction_hind_right" class="bkvb/abduction"/>
        <geom class="bkvb/abduction" pos="0 0.000111373 0.0031" quat="1 1 0 0"/>
        <body name="upper_leg_hind_right" pos="0.03085 0 -0.0648" quat="0 1 0 1">
          <inertial pos="-0.0241393 0.00324567 0.0453036" quat="-0.00604983 0.756969 -0.0854547 0.64781" mass="1.155"
            diaginertia="0.00563107 0.00519539 0.00126472"/>
          <joint name="hip_hind_right" class="bkvb/hip"/>
          <geom class="bkvb/upper_leg" pos="0.0679 -0.000111373 -0.03085" quat="1 -1 -1 -1"/>
          <geom class="bkvb/upper_leg_right" pos="0 0 0.05075" quat="0 0 -1 0"/>
          <body name="lower_leg_4" pos="-0.19 0 0.069575" quat="0 0 -1 0">
            <inertial pos="-0.0895493 -0.0301957 -3.02082e-08" quat="-0.101465 0.699789 0.101465 0.699789"
              mass="0.171238" diaginertia="0.00137406 0.00135746 3.05521e-05"/>
            <joint name="knee_hind_right" class="bkvb/knee"/>
            <geom class="bkvb/lower_leg" pos="-0.0649838 0.178542 0" quat="0.819152 0 0 -0.573576"/>
            <geom class="bkvb/foot" pos="-0.0649838 0.178542 0" quat="0.819152 0 0 -0.573576"/>
            <site name="foot_hind_right" class="bkvb/foot"/>
          </body>
        </body>
      </body>
    </body>
  </worldbody>

  <actuator>
    <general name="abduction_front_left" class="bkvb/abduction" joint="abduction_front_left"/>
    <general name="hip_front_left" class="bkvb/hip" joint="hip_front_left"/>
    <general name="knee_front_left" class="bkvb/knee" joint="knee_front_left"/>
    <general name="abduction_hind_left" class="bkvb/abduction" joint="abduction_hind_left"/>
    <general name="hip_hind_left" class="bkvb/hip" joint="hip_hind_left"/>
    <general name="knee_hind_left" class="bkvb/knee" joint="knee_hind_left"/>
    <general name="abduction_front_right" class="bkvb/abduction" joint="abduction_front_right"/>
    <general name="hip_front_right" class="bkvb/hip" joint="hip_front_right"/>
    <general name="knee_front_right" class="bkvb/knee" joint="knee_front_right"/>
    <general name="abduction_hind_right" class="bkvb/abduction" joint="abduction_hind_right"/>
    <general name="hip_hind_right" class="bkvb/hip" joint="hip_hind_right"/>
    <general name="knee_hind_right" class="bkvb/knee" joint="knee_hind_right"/>
  </actuator>

  <sensor>
    <jointpos joint="abduction_front_left" name="abduction_front_left_pos"/>
    <jointpos joint="hip_front_left" name="hip_front_left_pos"/>
    <jointpos joint="knee_front_left" name="knee_front_left_pos"/>
    <jointpos joint="abduction_hind_left" name="abduction_hind_left_pos"/>
    <jointpos joint="hip_hind_left" name="hip_hind_left_pos"/>
    <jointpos joint="knee_hind_left" name="knee_hind_left_pos"/>
    <jointpos joint="abduction_front_right" name="abduction_front_right_pos"/>
    <jointpos joint="hip_front_right" name="hip_front_right_pos"/>
    <jointpos joint="knee_front_right" name="knee_front_right_pos"/>
    <jointpos joint="abduction_hind_right" name="abduction_hind_right_pos"/>
    <jointpos joint="hip_hind_right" name="hip_hind_right_pos"/>
    <jointpos joint="knee_hind_right" name="knee_hind_right_pos"/>
    <jointvel joint="abduction_front_left" name="abduction_front_left_vel"/>
    <jointvel joint="hip_front_left" name="hip_front_left_vel"/>
    <jointvel joint="knee_front_left" name="knee_front_left_vel"/>
    <jointvel joint="abduction_hind_left" name="abduction_hind_left_vel"/>
    <jointvel joint="hip_hind_left" name="hip_hind_left_vel"/>
    <jointvel joint="knee_hind_left" name="knee_hind_left_vel"/>
    <jointvel joint="abduction_front_right" name="abduction_front_right_vel"/>
    <jointvel joint="hip_front_right" name="hip_front_right_vel"/>
    <jointvel joint="knee_front_right" name="knee_front_right_vel"/>
    <jointvel joint="abduction_hind_right" name="abduction_hind_right_vel"/>
    <jointvel joint="hip_hind_right" name="hip_hind_right_vel"/>
    <jointvel joint="knee_hind_right" name="knee_hind_right_vel"/>
    <gyro site="imu_frame" name="gyro"/>
    <accelerometer site="imu_frame" name="accelerometer"/>
    <framequat objtype="site" objname="imu_frame" name="orientation"/>
    <framepos objtype="site" objname="imu_frame" name="global_position"/>
    <framelinvel objtype="site" objname="imu_frame" name="global_linvel"/>
    <frameangvel objtype="site" objname="imu_frame" name="global_angvel"/>
  </sensor>
</mujoco>
"""