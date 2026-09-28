#!/usr/bin/env bash
# Reproduce all synthetic-benchmark certificates (needs numpy, scipy, mpmath 1.3.0).
set -e
cd "$(dirname "$0")"
mkdir -p routes certificates
# Stage 1: CMA-ES over Bezier curves (BERS stage 1 re-implementation)
python3 -W ignore run_stage1.py circular doublegyre techy
python3 -W ignore run_stage1_multi.py fourvortices
python3 -W ignore run_stage1_multi.py swirlys
# Stage 2: stationary points (full space; transverse family for travel time)
for n in circular techy swirlys; do python3 -W ignore refine.py $n; done
for n in circular fourvortices doublegyre techy; do python3 -W ignore refine_family.py $n; done
# 150-bit polishing
for n in circular fourvortices doublegyre techy; do python3 -W ignore polish_hp.py $n family; done
python3 -W ignore polish_hp.py swirlys full
# Certificates
for n in circular fourvortices doublegyre techy; do python3 -W ignore certify.py $n family; done
python3 -W ignore certify.py swirlys full
python3 -W ignore certify_full_hp.py circular
python3 -W ignore certify_full_hp.py techy
python3 -W ignore selftest.py
