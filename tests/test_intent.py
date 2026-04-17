"""Tests for the rule-based intent classifier."""

from app.retrieval.intent import classify, scripted_reply


def test_greeting():
    assert classify("hi") == "greeting"
    assert classify("hello there") == "greeting"
    assert classify("Good morning!") == "greeting"
    assert classify("salaam alaikum") == "greeting"


def test_goodbye():
    assert classify("bye") == "goodbye"
    assert classify("Thank you so much") == "goodbye"
    assert classify("shukran jazeelan") == "goodbye"


def test_signup():
    assert classify("I want to sign up") == "signup"
    assert classify("how do I register") == "signup"
    assert classify("I want to open an account") == "signup"
    assert classify("get me started") == "signup"


def test_signup_beats_greeting():
    assert classify("hi i want to sign up") == "signup"


def test_balance_check():
    assert classify("what is my balance") == "balance_check"
    assert classify("show me my portfolio") == "balance_check"


def test_unknown():
    assert classify("the weather in Tokyo today") == "unknown"
    assert classify("") == "unknown"


def test_scripted_reply_covers_all_scripted():
    assert scripted_reply("greeting") is not None
    assert scripted_reply("goodbye") is not None
    assert scripted_reply("signup") is not None
    assert scripted_reply("balance_check") is not None
    assert scripted_reply("unknown") is None
