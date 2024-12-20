import random

pizza_toppings = [
    "cheese", "tomato", "pepperoni", "mushroom", "sausage", "onion", "ham", "bacon", "chicken",
    "pineapple", "olive", "spinach", "pepper", "garlic", "anchovy", "basil", "beef", "artichoke",
    "jalapeno", "broccoli", "salami", "chorizo", "zucchini", "eggplant", "capers", "parsley",
    "arugula", "cilantro", "asparagus", "truffle", "caramelized", "ricotta", "mozzarella",
    "parmesan", "feta", "gorgonzola", "provolone", "prosciutto", "lobster", "shrimp", "clams",
    "corn", "kale", "potato", "tofu", "tempeh", "sun-dried", "turkey", "pesto",
    "fig", "beetroot", "cranberry", "apple", "pear", "mango", "coconut", "tarragon", "sage", "rosemary",
    "curry", "harissa", "sriracha", "honey", "balsamic", "bbq", "mustard", "chimichurri", "teriyaki",
    "hoisin", "tahini", "sour cream", "yogurt", "ranch", "blue cheese", "sriracha mayo", "wasabi",
    "dijon mustard", "avocado", "poppy seeds", "sesame seeds", "chia seeds", "paprika", "cumin",
    "coriander", "smoked salmon", "caviar", "fried egg", "scrambled egg", "brie", "cheddar", "camembert",
    "stilton", "cream cheese", "blueberry", "watermelon", "grape", "peach", "apricot", "pomegranate",
    "melon", "quinoa", "brown rice", "bulgur", "fennel", "leek", "radish", "pumpkin", "sweet potato",
    "sweet corn", "sour cherry", "blackberry", "strawberry", "raspberry", "blue cheese dressing",
    "goat cheese", "vegan cheese", "almonds", "cashews", "pine nuts", "walnuts", "pecans", "sunflower seeds",
    "pumpkin seeds", "crispy onions", "toasted coconut", "spinach artichoke", "tomato pesto", "sesame oil",
    "almond butter", "hummus", "baba ghanoush", "tandoori", "roasted garlic", "chimichurri sauce", "yuzu"
]

quantity_words = [
    "extra", "few", "more", "less", "many", "some", "any", "several", "all", "none", "much",
    "little", "enough", "plenty", "most", "least", "fewer", "scarce", "abundant", "additional",
    "double", "triple", "half", "multiple", "countless", "numerous", "ample", "minimal", "infinite",
    "fewest", "overabundant", "dozen", "pair", "single", "double", "whole", "one", "two", "three",
    "four", "five", "six", "seven", "eight", "nine", "ten", "hundred", "thousand", "million",
    "billion", "trillion", "quart", "gallon", "liter", "ounce", "pound", "inch", "foot", "yard",
    "mile", "centimeter", "kilometer", "metric ton", "short ton", "long ton", "cup", "teaspoon",
    "tablespoon", "dash", "pinch", "quart", "liter", "gill", "grain", "clove", "sprinkle", "handful",
    "heap", "load", "batch", "set", "group", "portion", "fragment", "segment", "slice", "piece",
    "bit", "amount", "quantity", "volume", "degree", "proportion", "count", "total", "sum",
    "aggregate", "spread", "extent", "scope", "range", "spectrum", "level", "division",
    "section", "slice", "zone", "tier", "segment", "piles", "clusters", "part", "subdivision",
    "segment", "quantity", "mass", "fraction", "ratio", "density", "accumulation",
    "a lot of", "a little of", "a few of", "a great deal of", "a bunch of", "a handful of",
    "a ton of", "a couple of", "a majority of", "a minority of", "a fraction of", "a portion of",
    "an abundance of", "a large number of", "a small amount of", "an excess of", "a surplus of",
    "a number of", "a pile of", "a heap of", "a load of", "a great number of", "a vast amount of"
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
    "sprite", "coke", "pepsi", "fanta", "lemonade", "milkshake", "smoothie", "milk", "beer", "wine",
    "whiskey", "vodka", "gin", "rum", "tequila", "margarita", "martini", "sangria", "mojito", "apple cider",
    "orange juice", "grape juice", "cranberry juice", "tomato juice", "pineapple juice", "lemon juice",
    "ginger ale", "iced tea", "hot tea", "coffee", "espresso", "latte", "cappuccino", "macchiato", "americano",
    "mocha", "frappuccino", "chai", "iced coffee", "hot chocolate", "milk tea", "kombucha", "water", "sparkling water",
    "coconut water", "energy drink", "sports drink", "root beer", "ginger beer", "iced lemonade", "shandy", "seltzer",
    "arizona tea", "green tea", "herbal tea", "peach iced tea", "berry smoothie", "green smoothie", "protein shake",
    "wheatgrass juice", "carrot juice", "beetroot juice", "acai bowl", "pomegranate juice", "banana milkshake",
    "vanilla milkshake", "chocolate milkshake", "strawberry milkshake", "blueberry smoothie", "mango smoothie",
    "watermelon juice", "soursop juice", "papaya juice", "grapefruit juice", "kiwi juice", "coconut milk",
    "soy milk", "almond milk", "oat milk", "rice milk", "cashew milk", "chocolate milk", "ice water", "sparkling lemonade",
    "lemon-lime soda", "cream soda", "apple soda", "berry soda", "fruit punch", "cocktail", "bloody mary", "screwdriver",
    "cosmopolitan", "long island iced tea", "bloody caesar", "daiquiri", "caipirinha", "cuba libre", "zombie",
    "tequila sunrise", "blue lagoon", "mai tai", "bahama mama", "hurricane", "pina colada", "sex on the beach",
    "mudslide", "frozen margarita", "negroni", "aperol spritz", "bloody mary mix", "vodka soda", "rum punch",
    "gin and tonic", "whiskey sour", "old fashioned", "manhattan", "mint julep", "irish coffee", "hot toddy",
    "lemon drop martini", "peach bellini", "sazerac", "irish whiskey", "bourbon", "scotch", "rye whiskey"
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
