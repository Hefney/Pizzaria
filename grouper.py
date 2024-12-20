from typing import List, Tuple
from tqdm import tqdm

def group_tokens_and_labels(tokens: List[List[str]],
                            labels: List[List[int]],
                            is_start_label: callable,
                            belongs_to_same_group: callable) -> Tuple[List[List[List[str]]], List[List[int]]]:
    """
    Group tokens and labels based on start labels and group membership.

    :param tokens: A list of lists, where each sublist contains tokens of a sentence.
    :param labels: A list of lists, where each sublist contains integer labels of a sentence.
    :param is_start_label: A function that takes a label and returns True if it's a _S label.
    :param belongs_to_same_group: A function that takes two labels (start, current) and determines if they belong to the same group.
    :return: Two lists: one with grouped tokens and one with corresponding labels.
    """
    grouped_tokens = []
    grouped_labels = []

    for sentence_tokens, sentence_labels in tqdm(zip(tokens, labels), desc='Group tokens and labels', total=len(tokens)):
        sentence_grouped_tokens = []
        sentence_grouped_labels = []
        current_group_tokens = []
        current_group_label = None

        for token, label in zip(sentence_tokens, sentence_labels):
            if is_start_label(label):
                # Start a new group if there's an active group
                if current_group_tokens:
                    sentence_grouped_tokens.append(current_group_tokens)
                    sentence_grouped_labels.append(current_group_label)

                # Start a new group
                current_group_tokens = [token]
                current_group_label = label
            elif current_group_label is not None and belongs_to_same_group(current_group_label, label):
                # Add to the current group
                current_group_tokens.append(token)
            else:
                # Finalize the current group if tokens don't belong
                if current_group_tokens:
                    sentence_grouped_tokens.append(current_group_tokens)
                    sentence_grouped_labels.append(current_group_label)

                # Reset the current group
                current_group_tokens = []
                current_group_label = None

        # Finalize any remaining group
        if current_group_tokens:
            sentence_grouped_tokens.append(current_group_tokens)
            sentence_grouped_labels.append(current_group_label)

        grouped_tokens.append(sentence_grouped_tokens)
        grouped_labels.append(sentence_grouped_labels)

    return grouped_tokens, grouped_labels


def reverse_grouping(grouped_data, get_cont_label: callable) -> Tuple[List[List[str]], List[List[int]]]:
    """
    Reverse the grouping of tokens and labels into flat sentences and labels.

    :param grouped_data: A list of grouped tokens and their labels.
    :param get_cont_label: A function that takes a start label and returns the corresponding continuation label.
    :return: A tuple containing flat sentences and their labels.
    """
    tokens = []
    labels = []

    for groups, group_labels in tqdm(grouped_data, desc='UnGroup tokens and labels', total=len(tokens)):
        sentence_tokens = []
        sentence_labels = []
        for group_tokens, start_label in zip(groups, group_labels):
            cont_label = get_cont_label(start_label)

            sentence_tokens.extend(group_tokens)
            sentence_labels.extend([start_label] + [cont_label] * (len(group_tokens) - 1))

        tokens.append(sentence_tokens)
        labels.append(sentence_labels)

    return tokens, labels


# Example usage
# Helper functions
def is_start_label(label: int) -> bool:
    # Example implementation: _S labels are even numbers
    return label % 2 == 0


def belongs_to_same_group(start_label: int, current_label: int) -> bool:
    # Example implementation: labels in the same group differ by 1
    return current_label == start_label + 1


def get_cont_label(start_label: int) -> int:
    # Example implementation: continuation label is start label + 1
    return start_label + 1


if __name__ == '__main__':   # small test to validate it works :)
    # Input data
    tokens = [["This", "is", "a", "test", "sentence"], ["Another", "example", "here"]]
    labels = [[2, 3, 4, 2, 3], [4, 5, 6]]

    grouped_tokens, grouped_labels = group_tokens_and_labels(tokens, labels, is_start_label, belongs_to_same_group)
    print("Grouped Tokens:", grouped_tokens)
    print("Grouped Labels:", grouped_labels)

    reversed_tokens, reversed_labels = reverse_grouping(list(zip(grouped_tokens, grouped_labels)), get_cont_label)
    print("Reversed Tokens:", reversed_tokens)
    print("Reversed Labels:", reversed_labels)
