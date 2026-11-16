"""League reinforcement learning on top of the behaviour-cloned temporal actor.

``incremental`` caches per-game encoder/attention state so a live decision
costs only its new records; ``sampling`` draws structured actions with exact
log-probabilities; ``ppo`` holds the value head, GAE/UPGO and the PPO loss;
``farm`` runs native games in worker processes against a batched GPU server;
``league`` samples opponents by PFSP; ``train`` is the entrypoint.
"""
