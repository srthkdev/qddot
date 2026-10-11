from bench.grade import add_false_done, claimed_success


def test_claim_parser_handles_negations():
    assert claimed_success("Done, it passes")
    assert not claimed_success("Not done yet")
    assert not claimed_success("It does not work")
    assert not claimed_success("I can't claim success")
    result = add_false_done({"success": False}, "The controller works")
    assert result["false_done"] is True
