"""LeRobot plugin registration and strict, station-local configuration."""

from dataclasses import dataclass, field

from lerobot.cameras import CameraConfig
from lerobot.robots import RobotConfig

from .protocol import finite


def topic(value: str) -> None:
    if not isinstance(value, str) or not value.startswith("/") or len(value) < 2:
        raise ValueError("ROS topic must be an absolute nonempty name")
    if any(c.isspace() for c in value):
        raise ValueError("ROS topic must not contain whitespace")


def positive(value: object, name: str) -> float:
    result = finite(value, name)
    if result <= 0:
        raise ValueError(f"{name} must be positive")
    return result


@CameraConfig.register_subclass("onero_ros_compressed")
@dataclass
class OneroRosCameraConfig(CameraConfig):
    topic: str = "/head/camera/rgb"
    width: int = 1600
    height: int = 1200
    fps: int = 30
    ros_domain_id: int = 0
    timeout_s: float = 5.0
    max_age_s: float = 0.2

    def __post_init__(self):
        topic(self.topic)
        for name in ("width", "height", "fps"):
            if type(getattr(self, name)) is not int or getattr(self, name) <= 0:
                raise ValueError(f"camera {name} must be a positive integer")
        if type(self.ros_domain_id) is not int or not 0 <= self.ros_domain_id <= 232:
            raise ValueError("ros_domain_id must be an integer in [0, 232]")
        positive(self.timeout_s, "camera timeout_s")
        positive(self.max_age_s, "camera max_age_s")


def default_cameras() -> dict[str, OneroRosCameraConfig]:
    return {
        name: OneroRosCameraConfig(topic=f"/{name}/camera/rgb")
        for name in ("head", "left", "right")
    }


@RobotConfig.register_subclass("onero_a1_h1")
@dataclass
class OneroRobotConfig(RobotConfig):
    model: str = "a1"
    ros_domain_id: int = 0
    left_joint_state_topic: str = "/left_joint_states"
    right_joint_state_topic: str = "/right_joint_states"
    left_gripper_state_topic: str = "/left_gripper_state"
    right_gripper_state_topic: str = "/right_gripper_state"
    joint_command_topic: str = "/record_data"
    gripper_command_topic: str = "/joystick_info"
    read_only: bool = True
    connect_timeout_s: float = 5.0
    max_state_age_s: float = 0.2
    command_timeout_s: float = 0.5
    cameras: dict[str, OneroRosCameraConfig] = field(default_factory=default_cameras)

    def __post_init__(self):
        super().__post_init__()
        if self.model not in ("a1", "h1") or type(self.read_only) is not bool:
            raise ValueError("model must be a1/h1 and read_only must be boolean")
        if type(self.ros_domain_id) is not int or not 0 <= self.ros_domain_id <= 232:
            raise ValueError("ros_domain_id must be an integer in [0, 232]")
        names = [getattr(self, n) for n in self.__dataclass_fields__ if n.endswith("_topic")]
        for name in names:
            topic(name)
        if len(set(names)) != len(names):
            raise ValueError("state and command topics must be distinct")
        for name in (
            "connect_timeout_s",
            "max_state_age_s",
            "command_timeout_s",
        ):
            positive(getattr(self, name), name)
        if set(self.cameras) != {"head", "left", "right"}:
            raise ValueError("exactly head, left and right cameras are required")
        for camera in self.cameras.values():
            if not isinstance(camera, OneroRosCameraConfig):
                raise ValueError("cameras must use onero_ros_compressed")
            if camera.ros_domain_id != self.ros_domain_id:
                raise ValueError("camera and robot ROS domains must match")
