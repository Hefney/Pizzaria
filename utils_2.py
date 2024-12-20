import math
import random

import extra_sets
import nltk
from nltk.stem import PorterStemmer
from tqdm import tqdm

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

    return vocab


def get_vocab(vocab_paths: list[str]):
    vocab = {}
    for path in vocab_paths:
        content = load_file_as_set(path)
        vocab_insert(vocab, content)

    vocab_insert(vocab, set(extra_sets.pizza_toppings))
    vocab_insert(vocab, set(extra_sets.quantity_words))
    vocab_insert(vocab, set(extra_sets.drinks))
    vocab_insert(vocab, set(extra_sets.extra_tokens))
    return vocab


def tokenize(text: str) -> list[str]:
    return [token for token in text.split(' ') if token != '']


def ValueSwap(value: str, options: list[str], sawp_prop=0.0, allow_unk=False):
    if random.random() < sawp_prop:
        if allow_unk:
            if random.random() < 1 / (1 + len(options)):
                return "<unk>"
        return random.choice(options)
    return value


clean_prefix = []
clean_suffixes = ["'d", ',', '.', "\'ll"]


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


def randomize_tokens(tokens: list[str], swap_prop=0.0):
    for index, _ in enumerate(tokens):
        if extra_sets.isTopping(tokens[index]):
            tokens[index] = ValueSwap(tokens[index], extra_sets.pizza_toppings, swap_prop, allow_unk=True)
        elif extra_sets.isQuantity(tokens[index]):
            tokens[index] = ValueSwap(tokens[index], extra_sets.quantity_words, swap_prop, allow_unk=False)
        elif extra_sets.isDrink(tokens[index]):
            tokens[index] = ValueSwap(tokens[index], extra_sets.drinks, swap_prop, allow_unk=True)
    return tokens


def preprocess_tokens(tokens: list[str], swap_prop=0.0):
    for index, _ in enumerate(tokens):
        tokens[index] = cleanWord(tokens[index])
        if extra_sets.isPersonalPronoun(tokens[index]):
            tokens[index] = "<pron>"
        elif extra_sets.isNumber(tokens[index]):
            tokens[index] = "<num>"
        elif extra_sets.isTopping(tokens[index]):
            tokens[index] = ValueSwap(tokens[index], extra_sets.pizza_toppings, swap_prop, allow_unk=True)
        elif extra_sets.isQuantity(tokens[index]):
            tokens[index] = ValueSwap(tokens[index], extra_sets.quantity_words, swap_prop, allow_unk=False)
        elif extra_sets.isDrink(tokens[index]):
            tokens[index] = ValueSwap(tokens[index], extra_sets.drinks, swap_prop, allow_unk=True)
    return tokens


def check_in_vocab(token, vocab):
    token = preprocess_tokens([token])[0]
    token = stemmer.stem(token)
    return token in vocab


def project_tokens(text: list[str], vocab: dict[str, int]) -> list[int]:
    result = []
    for token in text:
        before = token
        token = stemmer.stem(token)
        if token in vocab:
            result.append(vocab[token])
        else:
            print(f"Word: \"{before}\" -> \"{token}\" not in vocab")
            print(f"isPersonalPronoun: {extra_sets.isPersonalPronoun(token)}")
            print(f"isNumber: {extra_sets.isNumber(token)}")
            print(f"isTopping: {extra_sets.isTopping(token)}")
            print(f"isQuantity: {extra_sets.isQuantity(token)}")
            result.append(vocab["<unk>"])
    return result


def load_data_file(vocab, tag_map, sentences_file, labels_file, balancer=None):
    sentences = []
    labels = []

    with open(sentences_file) as f:
        for sentence in tqdm(f.read().splitlines(), desc="Sentences loader"):
            tokens = tokenize(sentence.lower())
            tokens = preprocess_tokens(tokens, 0.2)
            sentences.append(tokens)

    with open(labels_file) as f:
        for sentence in tqdm(f.read().splitlines(), desc="Labels loader"):
            if sentence == '':
                print("Empty label")
                continue
            labels.append([tag_map[label] for label in sentence.split(' ') if label != ''])

    if balancer is not None:
        sentences, labels = balancer(sentences, labels, tag_map, vocab)

    final_sentences = []
    for x in tqdm(sentences, desc="Projector"):
        final_sentences.append(project_tokens(x, vocab))

    return final_sentences, labels, len(sentences)


def orders_balancer(sentences: list[list[str]], labels: list[list[int]], tags, vocab):
    pizza_drink = []
    pizza_only = []
    drinks_only = []

    for sentence, label in tqdm(zip(sentences, labels), desc="Order balancer - preload"):
        has_pizza = tags["PIZZAORDER_S"] in label
        has_drink = tags["DRINKORDER_S"] in label

        if has_pizza and has_drink:
            pizza_drink.append((sentence, label))
        elif has_pizza:
            pizza_only.append((sentence, label))
        elif has_drink:
            drinks_only.append((sentence, label))

    while (len(pizza_only) - len(drinks_only)) / len(sentences) > 0.1:
        # print("pizza is more than drinks by more than 10%")
        c = random.choice(drinks_only)
        c = (randomize_tokens(c[0], 0.5), c[1])
        drinks_only.append(c)

    while (len(drinks_only) - len(pizza_only)) / len(sentences) > 0.1:
        # print("pizza is more than drinks by more than 10%")
        c = random.choice(pizza_only)
        c = (randomize_tokens(c[0], 0.5), c[1])
        pizza_only.append(c)

    all_items = []
    all_items.extend(pizza_only)
    all_items.extend(drinks_only)
    all_items.extend(pizza_drink)

    output_sentence = []
    outputs_labels = []

    for o in tqdm(all_items, desc="Order balancer - finalize"):
        output_sentence.append(o[0])
        outputs_labels.append(o[1])

    return output_sentence, outputs_labels


def pizza_orders_balancer(sentences: list[list[str]], labels: list[list[int]], tags, vocab):
    styled = []
    complex_orders = []
    negated = []
    negated_complex = []
    remainder = []

    for sentence, label in tqdm(zip(sentences, labels), desc="Pizza balancer - preload"):
        added = False
        if tags["STYLE_S"] in label or tags["NOT_STYLE_S"] in label:
            styled.append((sentence, label))
            added = True

        if tags["COMPLEX_TOPPING_S"] in label:
            complex_orders.append((sentence, label))
            added = True

        if tags["NOT_COMPLEX_TOPPING_S"] in label or tags["NOT_STYLE_S"] in label or tags["NOT_TOPPING_S"] in label:
            negated.append((sentence, label))
            added = True

        if tags["NOT_COMPLEX_TOPPING_S"] in label:
            negated_complex.append((sentence, label))
            added = True

        if not added:
            remainder.append((sentence, label))

    while len(styled) != 0 and len(styled) / len(sentences) < 0.8:
        c = random.choice(styled)
        c = (randomize_tokens(c[0], 0.5), c[1])
        styled.append(c)

    while len(complex_orders) != 0 and len(complex_orders) / len(sentences) < 0.3:  # at least 30% of the orders has complex topping
        c = random.choice(complex_orders)
        c = (randomize_tokens(c[0], 0.5), c[1])
        complex_orders.append(c)

    while len(negated) != 0 and len(negated) / len(sentences) < 0.5:  # at least 50% of the orders has negated attrs
        c = random.choice(negated)
        c = (randomize_tokens(c[0], 0.5), c[1])
        negated.append(c)

    while len(negated_complex) != 0 and len(negated_complex) / len(sentences) < 0.5:  # at least 50% of the orders has negated complex topping
        c = random.choice(negated_complex)
        c = (randomize_tokens(c[0], 0.5), c[1])
        negated_complex.append(c)

    all_items = []
    all_items.extend(styled)
    all_items.extend(complex_orders)
    all_items.extend(negated)
    all_items.extend(negated_complex)
    all_items.extend(remainder)

    output_sentence = []
    outputs_labels = []

    for o in tqdm(all_items, desc="Pizza balancer - finalize"):
        output_sentence.append(o[0])
        outputs_labels.append(o[1])

    return output_sentence, outputs_labels


def randomizer_balancer(count: int):
    def _rb_internal(sentences, labels, tags, vocab):
        all_items = []

        for sentence, label in tqdm(zip(sentences, labels), desc="Randomizer balancer - preload"):
            for i in range(count):
                all_items.append((randomize_tokens(sentence, math.exp(-i) if i != 0 else 0), label))

        output_sentence = []
        outputs_labels = []

        for o in tqdm(all_items, desc="Randomizer balancer - finalize"):
            output_sentence.append(o[0])
            outputs_labels.append(o[1])

        return output_sentence, outputs_labels

    return _rb_internal
