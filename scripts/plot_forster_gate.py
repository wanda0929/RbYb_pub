#!/usr/bin/env python3
"""Compatibility entry point for plotting Figure 3 from its checked record."""

import json

from reproduce_forster_gate import OUT, _plot_result


if __name__ == "__main__":
    _plot_result(json.loads(OUT.read_text()))
