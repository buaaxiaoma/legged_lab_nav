#!/usr/bin/env python3
from __future__ import annotations

import argparse
import time

from unitree_sdk2py.core.channel import ChannelFactoryInitialize, ChannelPublisher
from unitree_sdk2py.idl.default import unitree_go_msg_dds__HeightMap_
from unitree_sdk2py.idl.unitree_go.msg.dds_ import HeightMap_


DEFAULT_TARGET = (1.0, 0.0)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Continuously publish a body-frame local target to rt/local_target_pos_b."
    )
    parser.add_argument("--domain-id", type=int, default=0)
    parser.add_argument("--interface", type=str, default="lo")
    parser.add_argument("--topic", type=str, default="rt/local_target_pos_b")
    parser.add_argument("--rate", type=float, default=20.0, help="Publish rate in Hz. Keep this above 10 Hz.")
    parser.add_argument(
        "--target",
        type=float,
        nargs=2,
        metavar=("X_B", "Y_B"),
        default=DEFAULT_TARGET,
        help="Body-frame local target position [x_b y_b].",
    )
    args = parser.parse_args()

    ChannelFactoryInitialize(args.domain_id, args.interface)

    publisher = ChannelPublisher(args.topic, HeightMap_)
    publisher.Init()

    msg = unitree_go_msg_dds__HeightMap_()
    msg.frame_id = "base_link"
    msg.width = 2
    msg.height = 1
    msg.resolution = 1.0
    msg.origin = [0.0, 0.0]
    msg.data = [float(args.target[0]), float(args.target[1])]

    publish_period = 1.0 / max(args.rate, 1.0e-3)

    print(f"Publishing local target to {args.topic}")
    print(f"target_b=[{msg.data[0]:.3f}, {msg.data[1]:.3f}]")
    print("Press Ctrl+C to stop.")

    while True:
        msg.stamp = time.time()
        publisher.Write(msg)
        time.sleep(publish_period)


if __name__ == "__main__":
    main()
