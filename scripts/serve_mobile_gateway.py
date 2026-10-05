#!/usr/bin/env python3
from __future__ import annotations

import argparse
import os

from embodied_agent.integrations.mobile import GatewayWebsocketServer, build_pi05_mobile_gateway


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Serve the mobile PI05 embodied-agent gateway")
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8040)
    parser.add_argument("--action-host", default="127.0.0.1")
    parser.add_argument("--action-port", type=int, default=8020)
    parser.add_argument("--progress-host", default="127.0.0.1")
    parser.add_argument("--progress-port", type=int, default=8030)
    parser.add_argument("--molmo-url", default="http://127.0.0.1:8766")
    parser.add_argument("--sam-url", default="http://127.0.0.1:8765")
    parser.add_argument("--done-threshold", type=float, default=0.6)
    parser.add_argument("--done-count", type=int, default=1)
    parser.add_argument("--llm-api-base", required=True)
    parser.add_argument("--llm-model", required=True)
    parser.add_argument("--llm-api-key", default=os.environ.get("LLM_API_KEY", "EMPTY"))
    parser.add_argument(
        "--debug-dir",
        default="logs/mobile_gateway",
        help="Dump RGB / SAM / Molmo / plan / PI05 action context. Empty disables dump.",
    )
    parser.add_argument(
        "--no-debug",
        action="store_true",
        help="Disable side-channel debug dumps.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    from embodied_agent import OpenAIChatModel
    from openpi_client.websocket_client_policy import WebsocketClientPolicy

    model = OpenAIChatModel(
        args.llm_model,
        api_base=args.llm_api_base,
        api_key=args.llm_api_key,
    )
    debug_dir = None if args.no_debug or not str(args.debug_dir).strip() else args.debug_dir
    gateway = build_pi05_mobile_gateway(
        action_client=WebsocketClientPolicy(args.action_host, args.action_port),
        progress_client=WebsocketClientPolicy(args.progress_host, args.progress_port),
        molmo_url=args.molmo_url,
        sam_url=args.sam_url,
        entity_model=model,
        done_threshold=args.done_threshold,
        done_count_threshold=args.done_count,
        debug_dir=debug_dir,
    )
    if debug_dir is not None:
        print(f"debug dump: {debug_dir}", flush=True)
    GatewayWebsocketServer(gateway, host=args.host, port=args.port).serve_forever()


if __name__ == "__main__":
    main()
