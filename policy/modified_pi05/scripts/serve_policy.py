"""Serve the released mobile action policy from a JAX or PyTorch checkpoint."""
import dataclasses
import logging

import tyro

from openpi.policies import policy_config
from openpi.serving.websocket_policy_server import WebsocketPolicyServer
from openpi.training import config


@dataclasses.dataclass
class Checkpoint:
    config: str = "pi05_mobile_atomic_4task_short_horizon_memory_stride3"
    dir: str = tyro.MISSING


@dataclasses.dataclass
class Args:
    policy: Checkpoint = dataclasses.field(default_factory=Checkpoint)
    host: str = "0.0.0.0"
    port: int = 8020
    default_prompt: str | None = None
    use_ema: bool = False
    device: str | None = None
    seg_representation: str = "mask"


def main(args: Args):
    if args.policy.config != "pi05_mobile_atomic_4task_short_horizon_memory_stride3":
        raise ValueError("Use serve_progress_evaluator.py for the progress evaluator.")
    policy = policy_config.create_trained_policy(
        config.get_config(args.policy.config), args.policy.dir,
        default_prompt=args.default_prompt, use_ema=args.use_ema,
        pytorch_device=args.device, seg_infer_representation=args.seg_representation,
    )
    WebsocketPolicyServer(policy=policy, host=args.host, port=args.port, metadata=policy.metadata).serve_forever()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, force=True)
    main(tyro.cli(Args))
