#!/usr/bin/env python3
from __future__ import annotations
import argparse, json
from b029_runtime import materialize_raw
from b029_stimuli import ROOT
if __name__=="__main__":
    parser=argparse.ArgumentParser(); parser.add_argument("--run-id",required=True); args=parser.parse_args()
    print(json.dumps(materialize_raw(ROOT/"runs"/args.run_id),ensure_ascii=False,indent=2))

