import random

pizza_toppings = [
    "cheese",
    "tomato",
    "pepperoni",
    "mushroom",
    "sausage",
    "onion",
    "ham",
    "bacon",
    "chicken",
    "pineapple",
    "olive",
    "spinach",
    "pepper",
    "garlic",
    "anchovy",
    "basil",
    "beef",
    "artichoke",
    "jalapeno",
    "broccoli",
    "salami",
    "chorizo",
    "zucchini",
    "eggplant",
    "capers",
    "parsley",
    "arugula",
    "cilantro",
    "asparagus",
    "truffle",
    "caramelized",
    "ricotta",
    "mozzarella",
    "parmesan",
    "feta",
    "gorgonzola",
    "provolone",
    "prosciutto",
    "lobster",
    "shrimp",
    "clams",
    "corn",
    "kale",
    "potato",
    "tofu",
    "tempeh",
    "sun-dried",
    "turkey",
    "pesto"
]

quantity_words = [
    "extra",
    "few",
    "more",
    "less",
    "many",
    "some",
    "any",
    "several",
    "all",
    "none",
    "much",
    "little",
    "enough",
    "plenty",
    "most",
    "least",
    "fewer",
    "scarce",
    "abundant",
    "additional",
    "double",
    "triple",
    "half",
    "multiple",
    "countless",
    "numerous",
    "ample",
    "minimal",
    "infinite",
    "fewest",
    "overabundant",
    "dozen",
    "pair",
    "single",
    "double",
    "whole"
]

numbers = [
    "one",
    "two",
    "three",
    "four",
    "five",
    "six",
    "seven",
    "eight",
    "nine",
    "ten",
    "eleven",
    "twelve",
    "thirteen",
    "fourteen",
    "fifteen",
    "sixteen",
    "seventeen",
    "eighteen",
    "nineteen",
    "twenty",
    "thirty",
    "forty",
    "fifty",
    "sixty",
    "seventy",
    "eighty",
    "ninety",
    "hundred",
    "thousand",
    "million",
    "billion"
]

personal_pronouns = [
    "i",
    "me",
    "we",
    "us",
    "you",
    "he",
    "him",
    "she",
    "they",
    "them"
]

extra_tokens = [
    "<pron>",
    "<num>",
    "<unk>",
    "<pad>"
]

drinks = [
    "sprite",
    "coke",
    "pepsi",
    "fanta",
    "lemonade",
    "milkshake",
    "smoothie",
    "milk",
    "beer",
    "wine",
    "whiskey",
    "vodka",
    "gin",
    "rum",
    "tequila",
    "margarita",
    "martini",
    "sangria",
    "mojito"
]

negations = [
    "hold on",
    "don't",
]


def isNumber(s: str) -> bool:
    s = s.lower()
    if s.isdigit():
        return True
    if s in numbers:
        return True
    return False


def isQuantity(s: str) -> bool:
    s = s.lower()
    return s in quantity_words


def isTopping(s: str) -> bool:
    s = s.lower()
    return s in pizza_toppings


def getRandomTopping() -> str:
    return random.choice(pizza_toppings)


def isPersonalPronoun(s: str) -> bool:
    s = s.lower()
    return s in personal_pronouns


def isDrink(s: str) -> bool:
    s = s.lower()
    return s in drinks
