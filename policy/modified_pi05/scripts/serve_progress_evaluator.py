"""Serve a standalone task-progress evaluator over the OpenPI WebSocket RPC."""

from __future__ import annotations

import dataclasses
import logging

import tyro

from openpi.policies import policy_config
from openpi.serving.websocket_policy_server import WebsocketPolicyServer
from openpi.training import config as training_config


@dataclasses.dataclass
class Args:
    config: str
    checkpoint_dir: str
    port: int = 8030
    host: str = "0.0.0.0"
    device: str | None = None
    seg_representation: str = "mask"
    use_ema: bool = False


def main(args: Args) -> None:
    config = training_config.get_config(args.config)
    evaluator = policy_config.create_progress_evaluator_policy(
        config,
        args.checkpoint_dir,
        pytorch_device=args.device,
        use_ema=args.use_ema,
        seg_infer_representation=args.seg_representation,
    )
    metadata = {
        **evaluator.metadata,
        "config_name": args.config,
        "seg_infer_representation": args.seg_representation,
    }
    logging.info(
        "Serving progress evaluator %s from %s on %s:%d",
        args.config,
        args.checkpoint_dir,
        args.host,
        args.port,
    )
    WebsocketPolicyServer(
        policy=evaluator,
        host=args.host,
        port=args.port,
        metadata=metadata,
    ).serve_forever()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, force=True)
    main(tyro.cli(Args))
