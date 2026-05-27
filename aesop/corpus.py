#!/usr/bin/env python3
"""Generate a richer learnable corpus for calculator-scale neural LMs."""

from __future__ import annotations

import argparse
import random
from pathlib import Path


NAMES = [
    "Ava", "Ben", "Cora", "Dina", "Eli", "Ella", "Finn", "Gus",
    "Hana", "Iris", "Jack", "Juno", "Kai", "Leo", "Lily", "Mia",
    "Noah", "Omar", "Pia", "Ruby", "Sam", "Sofia", "Theo", "Tom",
    "Uma", "Vera", "Wren", "Zoe",
]
HEROES = [
    "artist", "child", "clockmaker", "fox", "frog", "gardener", "kitten",
    "mapmaker", "mouse", "owl", "pilot", "pony", "puppy", "rabbit",
    "robot", "sailor", "star", "turtle", "witch",
]
TRAITS = [
    "bold", "brave", "bright", "careful", "clever", "curious", "gentle",
    "happy", "kind", "quiet", "shy", "sleepy", "thoughtful", "wild",
]
PLACES = [
    "attic", "bakery", "beach", "clock tower", "forest", "garden",
    "greenhouse", "harbor", "hill", "kitchen", "library", "meadow",
    "moonlit room", "park", "river", "school", "train station",
    "windy bridge", "workshop",
]
OBJECTS = [
    "blue kite", "brass key", "cloud jar", "glass compass", "glowing marble",
    "green cup", "little drum", "lost button", "map of dots", "moon spoon",
    "paper crown", "paint brush", "pocket ladder", "rain whistle",
    "red ball", "ribbon lantern", "silver bell", "soft blanket",
    "stone flute", "story seed", "threaded coin", "tiny boat",
    "wooden box", "yellow hat",
]
GOALS = [
    "bake a moon cake", "build a tower", "cross the path", "draw a map",
    "find a new game", "fix a toy", "help a friend", "learn a song",
    "light the lantern", "make a present", "open the gate",
    "plant a seed", "solve the riddle", "teach a bird to sing",
    "visit the pond", "wake the old clock",
]
PROBLEMS = [
    "a loud sound made everyone stop",
    "all the shadows pointed the wrong way",
    "the bell fell under a chair",
    "the box would not open",
    "the bridge looked too thin",
    "the cloud jar began to hum",
    "the crown was missing",
    "the drum made no sound",
    "the kite got stuck in a tree",
    "the little boat drifted too far",
    "the map folded itself into a bird",
    "the moon cake floated away",
    "the path was muddy",
    "the seed had no water",
    "the sky became gray",
    "the tower leaned to one side",
    "a tiny door appeared in the wall",
    "everyone forgot the secret word",
    "footprints crossed the ceiling",
    "the clock counted backward",
    "the compass pointed at a song",
    "the lantern would only glow for jokes",
    "the rain fell upward",
    "the river whispered a warning",
    "the train left without tracks",
]
HELPERS = [
    "Daddy", "Mommy", "a careful snail", "a friendly robot",
    "a kind bird", "a little rabbit", "a smiling farmer", "a teacher",
    "an old turtle", "the quiet moon", "a sleepy dragon", "a tiny librarian",
    "the wind", "their best friend",
]
ADVICE = [
    "ask before you run", "be patient", "count the small clues",
    "follow the warm light", "listen to your heart", "look closely",
    "share what you have", "take a deep breath", "try one small step",
    "turn the problem around", "work together",
]
ACTIONS = [
    "asked for help", "carried water in the cup", "followed tiny footprints",
    "gave everyone a turn", "held the rope tight", "looked under every leaf",
    "made a careful plan", "painted a bright sign", "pushed with both hands",
    "rang the silver bell", "shared the little toy", "sang a soft song",
    "told a true story", "tried again slowly", "turned the map upside down",
    "wrote one brave word",
]
RESULTS = [
    "the bell rang clearly", "the boat came home", "the box opened with a sigh",
    "the bridge felt strong", "the cloud jar became quiet",
    "the crown was found", "the drum went boom", "the friend began to smile",
    "the kite danced in the air", "the lantern woke up",
    "the map showed a safe road", "the moon cake came down",
    "the path was safe", "the seed had a drink", "the sun came back",
    "the tower stood straight",
    "a hidden stair appeared", "everyone remembered the secret word",
    "the clock ticked forward", "the door opened into a garden",
    "the lantern filled the room with stars", "the rain became music",
]
ENDINGS = [
    "excited for tomorrow", "full of giggles", "glad to be together",
    "happy", "proud", "ready to play", "sleepy and safe", "warm inside",
]
MORALS = [
    "asking for help was brave", "being kind made the day better",
    "even a strange problem could become a story",
    "friends made hard things easier", "listening saved the adventure",
    "patience helped more than rushing", "sharing made the game more fun",
    "small steps could solve big worries", "trying again was worth it",
    "a good question could open a locked door",
    "a plan was stronger after a friend changed it",
    "not every answer had to be loud",
]
OPENINGS = [
    "At sunrise",
    "Before breakfast",
    "Every Tuesday",
    "In a tiny town",
    "Near the old bridge",
    "On the first windy day of spring",
    "Once upon a time",
    "When the moon was round",
]


def story_a(rng: random.Random) -> str:
    opening = rng.choice(OPENINGS)
    name = rng.choice(NAMES)
    hero = rng.choice(HEROES)
    trait = rng.choice(TRAITS)
    place = rng.choice(PLACES)
    obj = rng.choice(OBJECTS)
    goal = rng.choice(GOALS)
    problem = rng.choice(PROBLEMS)
    helper = rng.choice(HELPERS)
    advice = rng.choice(ADVICE)
    action = rng.choice(ACTIONS)
    result = rng.choice(RESULTS)
    ending = rng.choice(ENDINGS)
    moral = rng.choice(MORALS)
    if opening == "Once upon a time":
        first = f"{opening}, there was a {trait} {hero} named {name}."
    else:
        first = f"{opening}, there was a {trait} {hero} named {name}."
    return (
        f"{first} "
        f"{name} lived near the {place} and loved a {obj}. "
        f"One day, {name} wanted to {goal}. "
        f"But {problem}. "
        f"{name} felt worried, but did not quit. "
        f"{helper} said, {advice}. "
        f"{name} {action}. "
        f"Soon, {result}. "
        f"Everyone felt {ending}. "
        f"From then on, {name} remembered that {moral}."
    )


def story_b(rng: random.Random) -> str:
    opening = rng.choice(OPENINGS)
    name = rng.choice(NAMES)
    friend = rng.choice([n for n in NAMES if n != name])
    place = rng.choice(PLACES)
    obj = rng.choice(OBJECTS)
    problem = rng.choice(PROBLEMS)
    action = rng.choice(ACTIONS)
    result = rng.choice(RESULTS)
    moral = rng.choice(MORALS)
    return (
        f"{opening}, {name} and {friend} found a {obj} in the {place}. "
        f"It looked ordinary until {problem}. "
        f"{name} wanted to run, but {friend} whispered, look closely. "
        f"Together they {action}. "
        f"Soon, {result}. "
        f"They laughed because the day had become a secret adventure. "
        f"That night, {name} wrote that {moral}."
    )


def story_c(rng: random.Random) -> str:
    opening = rng.choice(OPENINGS)
    name = rng.choice(NAMES)
    hero = rng.choice(HEROES)
    place = rng.choice(PLACES)
    obj = rng.choice(OBJECTS)
    helper = rng.choice(HELPERS)
    advice = rng.choice(ADVICE)
    result = rng.choice(RESULTS)
    ending = rng.choice(ENDINGS)
    return (
        f"{opening}, a {hero} named {name} kept a {obj} beside the {place}. "
        f"Every morning it made a tiny sound. "
        f"One morning it was silent. "
        f"{name} asked {helper} what to do. "
        f"{helper} said, {advice}. "
        f"So {name} waited, listened, and smiled. "
        f"Soon, {result}. "
        f"The {hero} felt {ending}, and the {obj} seemed almost magical."
    )


def story_d(rng: random.Random) -> str:
    name = rng.choice(NAMES)
    friend = rng.choice([n for n in NAMES if n != name])
    place = rng.choice(PLACES)
    obj = rng.choice(OBJECTS)
    problem = rng.choice(PROBLEMS)
    advice = rng.choice(ADVICE)
    result = rng.choice(RESULTS)
    return (
        f"When the {obj} vanished, {name} did not cry. "
        f"{friend} found three clues in the {place}: a warm stone, a bent leaf, and a quiet footprint. "
        f"The clues were strange because {problem}. "
        f"{name} remembered to {advice}. "
        f"After one careful try, {result}. "
        f"{friend} laughed and said the mystery had been hiding in plain sight."
    )


def story_e(rng: random.Random) -> str:
    name = rng.choice(NAMES)
    hero = rng.choice(HEROES)
    place = rng.choice(PLACES)
    obj = rng.choice(OBJECTS)
    helper = rng.choice(HELPERS)
    action = rng.choice(ACTIONS)
    ending = rng.choice(ENDINGS)
    return (
        f"{name} the {hero} carried a {obj} through the {place}. "
        f"Do you know where it goes? asked {helper}. "
        f"Not yet, said {name}, but I can listen. "
        f"So {name} {action}. "
        f"The answer was not big or shiny. It was a small path home. "
        f"By night, the {hero} felt {ending}."
    )


def story_f(rng: random.Random) -> str:
    name = rng.choice(NAMES)
    trait = rng.choice(TRAITS)
    place = rng.choice(PLACES)
    obj = rng.choice(OBJECTS)
    goal = rng.choice(GOALS)
    problem = rng.choice(PROBLEMS)
    result = rng.choice(RESULTS)
    return (
        f"In the {place}, {name} built a machine from a {obj}, two buttons, and a spoon. "
        f"The machine was supposed to {goal}, but instead {problem}. "
        f"{name} was {trait}, so the first mistake became a test. "
        f"The second mistake became a joke. "
        f"The third mistake finally helped: {result}. "
        f"Everyone asked {name} to write the directions down."
    )


TEMPLATES = [story_a, story_a, story_a, story_b, story_c]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, default=Path("data/neuralstories/input.txt"))
    parser.add_argument("--stories", type=int, default=50000)
    parser.add_argument("--seed", type=int, default=19)
    args = parser.parse_args()
    rng = random.Random(args.seed)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w", encoding="ascii") as f:
        for _ in range(args.stories):
            f.write(rng.choice(TEMPLATES)(rng))
            f.write("\n\n")


if __name__ == "__main__":
    main()
