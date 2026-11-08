"""Evolutionary / black-box policy search for the mjlab G1 velocity task.

    policy.py      a population of MLP (or linear) policies evaluated in one
                   batched GPU forward pass, plus observation normalization
    algorithms.py  ARS, OpenAI-ES, PGPE and CMA-ES (full and diagonal)
    env.py         environment construction shared by training and evaluation
"""
