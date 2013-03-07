

robot_template = """
<Robot name="ur10">
  <KinBody>
    <Body name="base_link" type="dynamic">
      <Translation>0.0  0.0  0.0</Translation>
    </Body>
%s
    <!-- set the transparency of every geometry in the KinBody-->
    <transparency>0.1</transparency>
  </KinBody>
%s
</Robot>
"""
manipulator_template = """
  <!-- Specifying the manipulator structure-->
  <Manipulator name="%s">
    <effector>%s</effector>   <!-- last link where end effector is attached-->
    <base>%s</base>           <!-- base link-->
    <Translation>0 0 0</Translation>
  </Manipulator>
"""

link_template = """
    <Body name="%s_link" type="dynamic">
      <offsetfrom>%s</offsetfrom>
      <Translation>%s %s %s</Translation>
      <rotationaxis>%s %s %s %s</rotationaxis>
    </Body>
"""

joint_template = """
    <Joint circular="false" name="%s_joint" type="hinge">
      <Body>%s</Body>
      <Body>%s</Body>
      <offsetfrom>%s</offsetfrom>
      <limitsdeg>-360 360</limitsdeg>
      <axis>0 0 1</axis>
      <maxvel>3</maxvel>
      <resolution>1</resolution>
    </Joint>
"""
