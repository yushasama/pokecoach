"""One human-vs-AI practice game on the native engine, with optional move coaching."""

from __future__ import annotations

import random
from typing import Any

from cg.game import battle_finish, battle_select, battle_start

from deepmew_ex.coach import labels
from deepmew_ex.coach.explain import explain
from deepmew_ex.coach.brains import Brain, ModelBrain

Payload = dict[str, Any]
MAX_AI_STEPS = 4000


class GameSession:
    def __init__(self, your_deck: list[int], ai_deck: list[int], seat: int, ai: Brain, coach: Brain | None, names: tuple[str, str]) -> None:
        if seat not in (0, 1):
            raise ValueError('seat must be 0 or 1')
        self.your_deck, self.ai_deck, self.seat, self.ai, self.coach = your_deck, ai_deck, seat, ai, coach
        self.deck_names = names
        self.observation: Payload | None = None
        self.last_view: Payload | None = None
        self.log: list[str] = []
        self.result: int | None = None
        self.moves = 0

    def start(self) -> None:
        for brain in (self.ai, self.coach):
            if hasattr(brain, 'reset'):
                brain.reset()
        decks = (self.your_deck, self.ai_deck) if self.seat == 0 else (self.ai_deck, self.your_deck)
        observation, start = battle_start(list(decks[0]), list(decks[1]))
        if observation is None:
            raise RuntimeError(f'engine rejected decks: errorPlayer={start.errorPlayer}, errorType={start.errorType}')
        self.observation = observation
        self.advance()

    def finish(self) -> None:
        try:
            battle_finish()
        except Exception:
            pass
        for brain in (self.ai, self.coach):
            if hasattr(brain, 'close'):
                brain.close()

    def observers(self) -> list[Brain]:
        return [brain for brain in (self.coach, self.ai) if brain is not None and hasattr(brain, 'history')]

    def advance(self) -> None:
        """Let the AI play until it is the human's decision or the game ends."""
        for _ in range(MAX_AI_STEPS):
            observation = self.observation
            current = observation.get('current') or {}
            if current.get('result', -1) != -1:
                self.result = int(current['result'])
                self.log.append('You win!' if self.result == self.seat else ('Draw.' if self.result == 2 else 'The AI wins.'))
                return
            for brain in self.observers():
                brain.observe(observation)
            if current.get('yourIndex') == self.seat:
                # Only the human seat's own observations reach the screen and the log.
                self.log.extend(labels.log_lines(observation, self.seat))
                self.last_view = observation
                if self.coach is not None:
                    self.coach.advise(observation)
                return
            selection = self.ai.act(observation)
            self.observation = battle_select(selection)
        raise RuntimeError('AI exceeded the per-turn step budget')

    def legal(self, selection: list[int]) -> bool:
        select = (self.observation or {}).get('select') or {}
        options = select.get('option') or []
        return (len(set(selection)) == len(selection) and select.get('minCount', 1) <= len(selection) <= select.get('maxCount', 1) and all(type(value) is int and 0 <= value < len(options) for value in selection))

    def play(self, selection: list[int]) -> None:
        if self.result is not None:
            raise ValueError('game is over')
        if not self.legal(selection):
            raise ValueError('illegal selection for the current prompt')
        if self.coach is not None:
            self.coach.commit(selection)
        self.moves += 1
        self.observation = battle_select(selection)
        self.advance()

    def play_coach(self) -> None:
        if self.coach is None or self.coach.advice is None:
            raise ValueError('no coach advice is available')
        self.play(self.coach.advice.pick)

    def view(self) -> Payload:
        observation = self.last_view or self.observation
        select = (observation.get('select') or {}) if self.result is None else {}
        options = [{'position': index, 'label': labels.option_label(observation, option), 'kind': labels.enum_text(labels.OptionType, option.get('type')), 'targets': labels.option_targets(observation, option, self.seat), 'cardId': labels.option_card_id(observation, option), 'detail': labels.option_detail(option)} for index, option in enumerate(select.get('option') or [])]
        coach = None
        if self.result is None and self.coach is not None and self.coach.advice is not None:
            advice = self.coach.advice
            labelled = {row['position']: row['label'] for row in options}
            coach = {'source': advice.source, 'pick': advice.pick, 'pickLabels': [labelled.get(position, str(position)) for position in advice.pick],
                     'why': explain(observation, advice.pick),
                     'ranked': [{'positions': list(positions), 'label': ' + '.join(labelled.get(position, str(position)) for position in positions) or 'Choose nothing', 'probability': round(probability, 4)} for positions, probability in advice.ranked[:4]]}
        return {
            'result': None if self.result is None else ('win' if self.result == self.seat else 'draw' if self.result == 2 else 'loss'),
            'seat': self.seat, 'decks': {'you': self.deck_names[0], 'ai': self.deck_names[1]}, 'aiBrain': self.ai.label,
            'prompt': labels.prompt(observation) if self.result is None else '',
            'context': labels.enum_text(labels.SelectContext, select.get('context')) if select else '',
            'minCount': select.get('minCount', 1), 'maxCount': select.get('maxCount', 1),
            'options': options, 'coach': coach,
            'board': labels.board(observation, self.seat), 'log': self.log[-60:], 'moves': self.moves,
        }


def random_seat() -> int:
    return random.randint(0, 1)
