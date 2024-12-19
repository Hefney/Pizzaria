import random

import extra_sets
import nltk
from nltk.stem import PorterStemmer

nltk.download('punkt')
stemmer = PorterStemmer()


def get_tags(tags_path):
    tag_map = {}
    with open(tags_path) as f:
        for i, t in enumerate(f.read().splitlines()):
            tag_map[t] = i
    return tag_map


def load_file_as_set(file_path):
    with open(file_path) as f:
        return set(f.read().splitlines())


def vocab_insert(vocab: dict[str, int], items: set[str]):
    for word in items:
        word = cleanWord(word)
        word = stemmer.stem(word)
        if word not in vocab and not extra_sets.isNumber(word) and not extra_sets.isPersonalPronoun(word):
            vocab[word] = len(vocab)


def get_vocab(vocab_paths: list[str]):
    vocab = {}
    for path in vocab_paths:
        content = load_file_as_set(path)
        vocab_insert(vocab, content)

    vocab_insert(vocab, set(extra_sets.pizza_toppings))
    vocab_insert(vocab, set(extra_sets.quantity_words))
    vocab_insert(vocab, set(extra_sets.extra_tokens))
    return vocab


def tokenize(text: str) -> list[str]:
    return [token for token in text.split(' ') if token != '']


def ValueSwap(value: str, options: list[str], sawp_prop=0.0):
    if random.random() < sawp_prop:
        return random.choice(options)
    return value


clean_prefix = []
clean_suffixes = ["'d", ',', '.']


def cleanWord(word: str) -> str:
    for suffix in clean_suffixes:
        if word.endswith(suffix):
            word = word[:-len(suffix)]
            break
    for prefix in clean_prefix:
        if word.startswith(prefix):
            word = word[len(prefix):]
            break
    return word


def preprocess_tokens(tokens: list[str], swap_prop=0.0):
    for index, _ in enumerate(tokens):
        tokens[index] = cleanWord(tokens[index])
        if extra_sets.isPersonalPronoun(tokens[index]):
            tokens[index] = "<pron>"
        elif extra_sets.isNumber(tokens[index]):
            tokens[index] = "<num>"
        elif extra_sets.isTopping(tokens[index]):
            tokens[index] = ValueSwap(tokens[index], extra_sets.pizza_toppings, swap_prop)
        elif extra_sets.isQuantity(tokens[index]):
            tokens[index] = ValueSwap(tokens[index], extra_sets.quantity_words, swap_prop)

        tokens[index] = stemmer.stem(tokens[index])  # at the end stem the token
    return tokens


def project_tokens(text: list[str], vocab: dict[str, int]) -> list[int]:
    result = []
    for token in text:
        if token in vocab:
            result.append(vocab[token])
        else:
            print(f"Word: {token} not in vocab")
            print(f"isPersonalPronoun: {extra_sets.isPersonalPronoun(token)}")
            print(f"isNumber: {extra_sets.isNumber(token)}")
            print(f"isTopping: {extra_sets.isTopping(token)}")
            print(f"isQuantity: {extra_sets.isQuantity(token)}")
            result.append(vocab["<unk>"])
    return result


def load_data_file(vocab, tag_map, sentences_file, labels_file):
    sentences = []
    labels = []

    with open(sentences_file) as f:
        for sentence in f.read().splitlines():
            tokens = tokenize(sentence.lower())
            tokens = preprocess_tokens(tokens, 0.2)
            sentences.append(project_tokens(tokens, vocab))

    with open(labels_file) as f:
        for sentence in f.read().splitlines():
            # replace each label by its index
            if sentence == '':
                print("Empty label")
                continue
            l = [tag_map[label] for label in sentence.split(' ') if label != '']  # I added plus 1 here
            labels.append(l)
    return sentences, labels, len(sentences)
