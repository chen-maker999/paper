from skc.alfworld import AlfWorldState, items, receptacles


def test_items_and_receptacles():
    assert items("a keychain 1, a watch 3, and a watch 2") == ["keychain 1", "watch 3", "watch 2"]
    assert items("nothing") == []
    obs = "You are in the middle of a room. Looking quickly around you, you see a bed 1, a desk 2, and a safe 1.\n\nYour task is to: x."
    assert receptacles(obs) == ["bed 1", "desk 2", "safe 1"]


def test_take_and_move_supersede_contents():
    s = AlfWorldState()
    w = s.update("go to sofa 1", "You arrive at sofa 1. On the sofa 1, you see a box 1, a keychain 2, and a remotecontrol 1.", 2)
    assert [x.value for x in w] == ["sofa 1 contains: box 1, keychain 2, remotecontrol 1"]
    w = s.update("take keychain 2 from sofa 1", "You pick up the keychain 2 from the sofa 1.", 4)
    vals = {x.key: x.value for x in w}
    assert vals["recep:sofa 1"] == "sofa 1 contains: box 1, remotecontrol 1"
    assert vals["inv"] == "you are carrying: keychain 2"
    s.update("go to safe 1", "You arrive at safe 1. The safe 1 is closed.", 6)
    assert s.recep_value("safe 1") == "safe 1 is closed, contents not seen"
    s.update("open safe 1", "You open the safe 1. The safe 1 is open. In it, you see a watch 3, and a watch 2.", 8)
    assert s.recep_value("safe 1") == "safe 1 (open) contains: watch 3, watch 2"
    w = s.update("move keychain 2 to safe 1", "You move the keychain 2 to the safe 1.", 10)
    vals = {x.key: x.value for x in w}
    assert vals["recep:safe 1"] == "safe 1 (open) contains: watch 3, watch 2, keychain 2"
    assert vals["inv"] == "you are carrying nothing"


def test_transformations_and_no_op():
    s = AlfWorldState()
    s.update("take pan 1 from stoveburner 3", "You pick up the pan 1 from the stoveburner 3.", 2)
    w = s.update("clean pan 1 with sinkbasin 1", "You clean the pan 1 using the sinkbasin 1.", 4)
    vals = {x.key: x.value for x in w}
    assert vals["obj:pan 1"] == "pan 1 has been cleaned"
    assert vals["inv"] == "you are carrying: pan 1 (cleaned)"
    assert s.update("go to moon 1", "Nothing happens.", 6) == []
    w = s.update("use desklamp 1", "You turn on the desklamp 1.", 8)
    assert w[0].value == "desklamp 1 has been turned on"
