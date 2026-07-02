"""Unit tests for the instruction parser:  pytest test/test_command_parser.py"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from lexicon.command_parser import parse_instruction, wants_nearest


def test_go_to_the_red_chair():
    assert parse_instruction("go to the red chair") == ["red chair", "chair"]


def test_find_me_the_nearest_mug():
    assert parse_instruction("find me the nearest mug") == ["mug"]


def test_where_is_the_backpack():
    assert parse_instruction("where is the backpack?") == ["backpack"]


def test_navigate_toward_a_person():
    assert parse_instruction("navigate toward a person") == ["person"]


def test_plain_noun_phrase():
    assert parse_instruction("blue water bottle") == ["blue water bottle", "bottle"]


def test_please_and_punctuation_stripped():
    assert parse_instruction("Please find the laptop.") == ["laptop"]


def test_empty_after_strip():
    assert parse_instruction("go to the") == []


def test_wants_nearest_detects_selector():
    assert wants_nearest("find the nearest mug")
    assert wants_nearest("closest chair")
    assert not wants_nearest("the red chair")
