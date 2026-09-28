import random
import re
from collections import Counter, defaultdict


class BigramModel:
    def __init__(self, corpus):
        text = " ".join(corpus)
        words = re.findall(r"\b\w+\b", text.lower())
        bigram_counts = Counter(zip(words[:-1], words[1:]))
        unigram_counts = Counter(words)
        self.bigram_probs = defaultdict(dict)
        for (w1, w2), count in bigram_counts.items():
            self.bigram_probs[w1][w2] = count / unigram_counts[w1]

    def generate_text(self, start_word, length=10):
        current = start_word.lower()
        words = [current]
        for _ in range(length - 1):
            next_words = self.bigram_probs.get(current)
            if not next_words:
                break
            current = random.choices(
                list(next_words.keys()), weights=list(next_words.values())
            )[0]
            words.append(current)
        return " ".join(words)
