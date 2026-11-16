"""DeepMew Coach: build a Pokémon TCG deck, then practice against a random AI with a coach.

    python -m deepmew_ex.coach.server [--port 8765] [--checkpoint PATH]

Opens http://127.0.0.1:<port>. Our models run on CPU so a training run can keep
the GPU. One game at a time (the native engine is process-global).

Coach: the strongest general agent available (the deck-agnostic 7th-place
league agent when downloaded, else our BC model). Opponents: picked at random
from our models, each piloting its own deck.
"""

from __future__ import annotations

import argparse
import json
import os
import random
import sys
import traceback
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path

from deepmew_ex.experiments.config import PROJECT_ROOT

sys.path.insert(0, str(PROJECT_ROOT / 'source'))

from deepmew_ex.coach import labels  # noqa: E402  (needs the engine path above)
from deepmew_ex.coach.brains import CoachModel, ExternalBrain, ExternalCoach, GreedyBrain, ModelBrain, RulesBrain, find_checkpoints, load_coach_model  # noqa: E402
from deepmew_ex.coach.decks import DeckStore, catalog, problems  # noqa: E402
from deepmew_ex.coach.session import GameSession  # noqa: E402

STATIC = Path(__file__).with_name('static')
STATIC_FILES = {'index.html': 'text/html; charset=utf-8', 'card_images.json': 'application/json'}
# A local download wins; otherwise the copy vendored in the pokecoach repo (third_party/, MIT).
EXTERNAL_7TH = next((path for path in (PROJECT_ROOT / 'outputs/external_agents/z3rox_7th_1dde9eca/kaggle_submission', PROJECT_ROOT / 'third_party/z3rox_7th/kaggle_submission') if (path / 'main.py').is_file()), PROJECT_ROOT / 'third_party/z3rox_7th/kaggle_submission')
# Local training runs first; models/coach holds the weights-only checkpoints shipped with the repo.
RUN_ROOTS = [PROJECT_ROOT / 'outputs/training-runs', PROJECT_ROOT / 'outputs/training-runs-archive', PROJECT_ROOT / 'models/coach']
SAVED_DECKS = PROJECT_ROOT / 'outputs/coach_decks'


class Coach:
    def __init__(self, checkpoint: Path | None) -> None:
        self.samples = self.sample_decks()
        self.decks = DeckStore(SAVED_DECKS, self.samples)
        candidates = [checkpoint] if checkpoint else find_checkpoints(RUN_ROOTS)
        self.model: CoachModel | None = None
        self.model_error: str | None = None
        for path in candidates:
            try:
                self.model = load_coach_model(path)
                break
            except Exception as error:
                self.model_error = f'{path.name}: {error}'
        self.has_seventh = (EXTERNAL_7TH / 'main.py').is_file()
        self.session: GameSession | None = None
        images = json.loads((STATIC / 'card_images.json').read_text(encoding='utf-8'))
        self.catalog = catalog({int(card) for card in images['cards']})

    @staticmethod
    def sample_decks() -> dict[str, list[int]]:
        paths = [PROJECT_ROOT / 'source/deck.csv', *sorted((PROJECT_ROOT / 'league/decks').glob('*.deck.csv')), EXTERNAL_7TH / 'deck.csv']
        decks: dict[str, list[int]] = {}
        for path in paths:
            try:
                deck = labels.read_deck(path)
            except (OSError, ValueError):
                continue
            name = 'Sample: ' + labels.deck_label(deck)
            decks[name if name not in decks else f'{name} ({path.stem[:6]})'] = deck
        return decks

    def opponents(self) -> list[tuple[str, str]]:
        """(brain, deck name) pairs a random opponent is drawn from; each AI pilots a deck it knows."""
        pool = [('rules', next(name for name, deck in self.samples.items() if deck == labels.read_deck(PROJECT_ROOT / 'source/deck.csv')))]
        if self.model is not None:
            pool += [('model', name) for name in self.samples]
        if self.has_seventh:
            pool.append(('seventh', next(name for name, deck in self.samples.items() if deck == labels.read_deck(EXTERNAL_7TH / 'deck.csv'))))
        return pool

    def coach_label(self) -> str:
        if self.has_seventh:
            return '7th-place league agent (deck-agnostic)'
        return f'our BC model ({self.model.presentations:,} decisions)' if self.model else 'none'

    def meta(self) -> dict[str, object]:
        decks = self.decks.all()
        return {'decks': [{'name': name, 'saved': row['saved'], 'size': len(row['cards'])} for name, row in decks.items()],
                'coach': self.coach_label(), 'opponents': sorted({brain for brain, _ in self.opponents()}), 'modelError': self.model_error}

    def brain(self, name: str):
        if name == 'model' and self.model is not None:
            return ModelBrain(self.model, 'DeepMew BC model')
        if name == 'seventh' and self.has_seventh:
            return ExternalBrain(EXTERNAL_7TH, PROJECT_ROOT / 'source', '7th-place league agent')
        return RulesBrain() if name == 'rules' else GreedyBrain()

    def make_coach(self):
        if self.has_seventh:
            return ExternalCoach(EXTERNAL_7TH, PROJECT_ROOT / 'source', '7th-place league agent')
        return ModelBrain(self.model, 'coach') if self.model is not None else None

    def deck(self, name: str) -> list[int]:
        decks = self.decks.all()
        if name not in decks:
            raise ValueError(f'unknown deck: {name}')
        return list(decks[name]['cards'])

    def new_game(self, body: dict[str, object]) -> dict[str, object]:
        yours = str(body.get('deck') or '')
        your_cards = self.deck(yours)
        issues = problems(your_cards)
        if issues:
            raise ValueError('Your deck is not legal: ' + '; '.join(issues))
        brain, their_deck = random.choice(self.opponents())
        seat = body.get('seat')
        seat = random.randint(0, 1) if seat not in (0, 1) else int(seat)
        if self.session is not None:
            self.session.finish()
        opponent = self.brain(brain)
        self.session = GameSession(your_cards, self.deck(their_deck), seat, opponent, self.make_coach() if body.get('coach', True) else None, (yours, their_deck.removeprefix('Sample: ')))
        self.session.start()
        return self.session.view()


def handler_for(coach: Coach) -> type[BaseHTTPRequestHandler]:
    class Handler(BaseHTTPRequestHandler):
        def log_message(self, format: str, *args: object) -> None:
            pass

        def send_json(self, value: object, status: int = 200) -> None:
            data = json.dumps(value).encode('utf-8')
            self.send_response(status)
            self.send_header('Content-Type', 'application/json; charset=utf-8')
            self.send_header('Content-Length', str(len(data)))
            self.end_headers()
            self.wfile.write(data)

        def do_GET(self) -> None:
            name = 'index.html' if self.path in ('/', '/index.html') else self.path.lstrip('/')
            if name in STATIC_FILES:
                data = (STATIC / name).read_bytes()
                self.send_response(200)
                self.send_header('Content-Type', STATIC_FILES[name])
                self.send_header('Content-Length', str(len(data)))
                self.end_headers()
                self.wfile.write(data)
            elif self.path == '/api/meta':
                self.send_json(coach.meta())
            elif self.path == '/api/cards':
                self.send_json(coach.catalog)
            elif self.path.startswith('/api/deck?name='):
                from urllib.parse import unquote
                name = unquote(self.path.split('=', 1)[1])
                try:
                    cards = coach.deck(name)
                    self.send_json({'name': name, 'cards': cards, 'problems': problems(cards)})
                except ValueError as error:
                    self.send_json({'error': str(error)}, 404)
            elif self.path == '/api/state':
                self.send_json(coach.session.view() if coach.session else {'idle': True})
            else:
                self.send_json({'error': 'not found'}, 404)

        def do_POST(self) -> None:
            try:
                body = json.loads(self.rfile.read(int(self.headers.get('Content-Length') or 0)) or b'{}')
                if self.path == '/api/decks/check':
                    self.send_json({'problems': problems([int(card) for card in body.get('cards', [])])})
                elif self.path == '/api/decks/save':
                    coach.decks.save(str(body.get('name') or ''), [int(card) for card in body.get('cards', [])])
                    self.send_json(coach.meta())
                elif self.path == '/api/decks/delete':
                    coach.decks.delete(str(body.get('name') or ''))
                    self.send_json(coach.meta())
                elif self.path == '/api/new':
                    self.send_json(coach.new_game(body))
                elif self.path == '/api/move' and coach.session:
                    coach.session.play([int(value) for value in body.get('positions', [])])
                    self.send_json(coach.session.view())
                elif self.path == '/api/coach-move' and coach.session:
                    coach.session.play_coach()
                    self.send_json(coach.session.view())
                else:
                    self.send_json({'error': 'no game in progress'}, 400)
            except ValueError as error:
                self.send_json({'error': str(error)}, 400)
            except Exception as error:
                traceback.print_exc()
                self.send_json({'error': f'{type(error).__name__}: {error}'}, 500)

    return Handler


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--port', type=int, default=int(os.environ.get('PORT', 8765)))
    parser.add_argument('--checkpoint', type=Path)
    args = parser.parse_args()
    coach = Coach(args.checkpoint)
    print(json.dumps({'url': f'http://127.0.0.1:{args.port}', 'coach': coach.coach_label(), 'opponents': coach.meta()['opponents'], 'cards': len(coach.catalog)}), flush=True)
    HTTPServer(('127.0.0.1', args.port), handler_for(coach)).serve_forever()


if __name__ == '__main__':
    main()
