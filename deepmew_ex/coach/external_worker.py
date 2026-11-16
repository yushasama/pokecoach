"""Isolated worker: runs a downloaded Kaggle agent; JSON obs in on stdin, JSON selection out on the real stdout."""
import json, os, sys
protocol = os.fdopen(os.dup(1), 'w')
sys.stdout = sys.stderr  # agent prints never corrupt the protocol
sys.path.insert(0, os.getcwd())
sys.path.insert(1, sys.argv[1])  # directory providing the competition `cg` package
from main import agent
for line in sys.stdin:
    try:
        reply = {'ok': [int(v) for v in agent(json.loads(line))]}
        # Agents that publish per-action scores (e.g. funcnet4_agent.LAST_SCORES) power the coach view.
        for module in ('funcnet4_agent',):
            scores = getattr(sys.modules.get(module), 'LAST_SCORES', None)
            if isinstance(scores, dict) and scores.get('actions'):
                reply['scores'] = {'actions': scores['actions'], 'scores': scores['scores'], 'override': bool(scores.get('override'))}
    except Exception as error:
        reply = {'error': f'{type(error).__name__}: {error}'}
    protocol.write(json.dumps(reply) + '\n'); protocol.flush()
