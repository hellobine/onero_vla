"""Private ROS context. ROS imports happen only when a connection is opened."""

import threading
import uuid


class RosConnection:
    def __init__(self, domain: int):
        from rclpy.context import Context
        from rclpy.executors import SingleThreadedExecutor
        from rclpy.node import Node
        from rclpy.qos import HistoryPolicy, QoSProfile, ReliabilityPolicy, qos_profile_sensor_data
        from sensor_msgs.msg import CompressedImage, JointState
        from std_msgs.msg import Float64MultiArray, Int32, UInt8

        self.context = Context()
        self.context.init(domain_id=domain)
        self.node = Node(f"onero_{uuid.uuid4().hex[:12]}", context=self.context)
        self.executor = SingleThreadedExecutor(context=self.context)
        self.executor.add_node(self.node)
        self.types = {
            "image": CompressedImage,
            "joints": JointState,
            "gripper": UInt8,
            "arm_command": Float64MultiArray,
            "gripper_command": Int32,
        }
        self.qos = QoSProfile(
            reliability=ReliabilityPolicy.RELIABLE, history=HistoryPolicy.KEEP_LAST, depth=1
        )
        self.image_qos = qos_profile_sensor_data
        self.error = None
        self.thread = None
        self.stopping = threading.Event()

    def subscribe(self, kind, topic, callback):
        self.node.create_subscription(
            self.types[kind], topic, callback, self.image_qos if kind == "image" else self.qos
        )

    def publisher(self, kind, topic):
        return self.node.create_publisher(self.types[kind], topic, self.qos)

    def publish(self, publisher, kind, data):
        message = self.types[kind]()
        message.data = data
        publisher.publish(message)

    def start(self):
        def spin():
            try:
                self.executor.spin()
                if not self.stopping.is_set():
                    self.error = RuntimeError("ROS executor exited")
            except BaseException as exc:
                self.error = exc

        self.thread = threading.Thread(target=spin, name="onero-ros", daemon=True)
        self.thread.start()

    def close(self):
        self.stopping.set()
        self.executor.shutdown(timeout_sec=1.0)
        if self.thread is not None:
            self.thread.join(timeout=1.0)
            if self.thread.is_alive():
                raise RuntimeError("ROS thread did not stop; connection teardown is incomplete")
        self.node.destroy_node()
        self.context.try_shutdown()
