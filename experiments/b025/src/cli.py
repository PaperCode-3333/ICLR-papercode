from __future__ import annotations

import argparse
import json

from src.benchmarks import bbeh, ifeval, ruler, socialeval, socialeval_iae

RUNNERS = {
    "ifeval": ifeval,
    "bbeh": bbeh,
    "ruler": ruler,
    "socialeval": socialeval,
    "socialeval_iae": socialeval_iae,
}


def main() -> None:
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="command", required=True)

    run = subparsers.add_parser("run")
    run.add_argument("--benchmark", choices=sorted(RUNNERS), required=True)
    run.add_argument("--run-id", required=True)
    run.add_argument("--model", required=True)
    run.add_argument("--endpoint")
    run.add_argument("--limit", type=int)
    run.add_argument("--mini", action="store_true")
    run.add_argument("--formal", action="store_true")

    score = subparsers.add_parser("score")
    score.add_argument("--benchmark", choices=sorted(RUNNERS), required=True)
    score.add_argument("--run-id", required=True)
    score.add_argument("--model", required=True)

    args = parser.parse_args()
    module = RUNNERS[args.benchmark]
    if args.command == "run":
        if args.formal and (args.limit is not None or args.mini):
            parser.error("Formal runs reject LIMIT and mini/debug datasets")
        kwargs = {
            "run_id": args.run_id,
            "model_key": args.model,
            "endpoint_override": args.endpoint,
            "limit": args.limit,
        }
        if args.benchmark == "bbeh":
            kwargs["mini"] = args.mini
        elif args.mini:
            parser.error("--mini is only valid for BBEH")
        result = module.generate(**kwargs)
    else:
        result = module.score(run_id=args.run_id, model_key=args.model)
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
