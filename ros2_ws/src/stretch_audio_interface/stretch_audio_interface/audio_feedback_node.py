#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from std_msgs.msg import String
import json
import subprocess
from TTS.api import TTS


class AudioFeedbackNode(Node):
    def __init__(self, template_file: str,
                 model: str = "tts_models/en/ljspeech/tacotron2-DDC",
                 player: str = "paplay"):
        super().__init__('audio_feedback_node')

        # load templates
        with open(template_file, "r", encoding="utf-8") as f:
            self.templates = json.load(f)

        # init TTS
        self.tts = TTS(model)
        self.player = player
        self.tmp_file = "/tmp/tts_output.wav"

        # subscriber
        self.sub_ = self.create_subscription(
            String, "audio_feedback", self.listener_callback, 10
        )

        self.get_logger().info("AudioFeedbackNode initialized.")

    def listener_callback(self, msg: String):
        self.get_logger().info(f"Received message: {msg.data}")
        try:
            payload = json.loads(msg.data)
            message_type = payload.get("message_type")
            if not message_type:
                return

            kwargs = {k: v for k, v in payload.items() if k != "message_type"}
            template = self.templates.get(message_type)
            if not template:
                self.get_logger().error(f"Unknown type {message_type}")
                return

            try:
                text = template.format(**kwargs)
            except KeyError as e:
                self.get_logger().error(f"Missing {e} for {message_type}")
                return

            # synthesize and play
            self.tts.tts_to_file(text=text, file_path=self.tmp_file)
            try:
                subprocess.run([self.player, self.tmp_file], check=True)
            except Exception as e:
                self.get_logger().warn(
                    f"Playback failed with {self.player}: {e}. "
                    f"Audio saved at {self.tmp_file}"
                )

        except Exception as e:
            self.get_logger().error(f"Invalid message: {e}")


def main(args=None):
    import sys
    rclpy.init(args=args)

    # accept template file arg, ignore ROS --ros-args
    template_file = None
    for arg in sys.argv[1:]:
        if not arg.startswith("--"):
            template_file = arg
            break

    if template_file is None:
        template_file = "templates.json"

    node = AudioFeedbackNode(template_file)
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()
