import SPELLCORRECTOR.spellcorrector as sc


try:
    test =  sc.SpellCorrector()
    ans = test.check("orn Based on author  dba Justin Cronin's trilogy of the same name, `The Passage' is a character-driven action drama that focuses on Project Noah, a secret medical facility where scientists experiment with a dangerous virus that could lead to the cure for all disease -- but it also could potentially wipe ")
    print(ans)
except TypeError:
    print("Check function takes in a string")