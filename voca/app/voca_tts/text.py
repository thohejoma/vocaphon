# SPDX-License-Identifier: Apache-2.0
"""Append-only bilingual token ABI; phonemization is provided by Phony, not eSpeak."""
PAD, BOS, EOS, SP, UNK = "<pad>", "<bos>", "<eos>", "<sp>", "<unk>"
SPECIALS = (PAD, BOS, EOS, SP, UNK)
PUNCTUATION = (",", ".", "?", "!", ";", ":", '"', "'", "(", ")", "—", "…")
CONSONANTS = ("p", "b", "t", "d", "k", "ɡ", "f", "v", "θ", "ð", "s", "z", "ʃ", "ʒ", "h",
              "m", "n", "ŋ", "l", "ɹ", "j", "w", "tʃ", "dʒ", "ɾ", "ʔ")
VOWELS = ("ə", "ɐ", "ɚ", "ᵻ", "ɪ", "i", "iː", "ɛ", "æ", "ʌ", "ɑː", "ɔ", "ɔː", "ʊ", "uː", "ɜː",
          "eɪ", "aɪ", "ɔɪ", "oʊ", "aʊ", "iə", "oː",
          "ɑːɹ", "ɔːɹ", "oːɹ", "ɪɹ", "ɛɹ", "ʊɹ", "aɪɚ", "aɪə", "əl", "n̩")
# German phones are appended without changing the English token IDs.
GERMAN_CONSONANTS = ("ʁ", "ç", "x", "ts", "pf")
GERMAN_VOWELS = ("a", "aː", "eː", "yː", "øː", "œ", "ʏ")
PHONES = CONSONANTS + VOWELS + GERMAN_CONSONANTS + GERMAN_VOWELS
# Append-only: changing the order or removing a symbol invalidates checkpoints.
SYMBOLS = SPECIALS + PUNCTUATION + PHONES
SYMBOL_TO_ID = {symbol: index for index, symbol in enumerate(SYMBOLS)}
PHONE_SET = frozenset(PHONES)
PHONE_IDS = frozenset(SYMBOL_TO_ID[p] for p in PHONES) | {SYMBOL_TO_ID[UNK]}
VOICELESS = frozenset(("p", "t", "k", "f", "θ", "s", "ʃ", "h", "tʃ", "ʔ", "ç", "x", "ts", "pf"))
VOWEL_SET = frozenset(VOWELS + GERMAN_VOWELS)
STRESS_MARKS = {"ˈ": 1, "ˌ": 2}

