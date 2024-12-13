#include <iostream>
#include <string>
#include <vector>
#include <sstream>
#include <map>
#include "nlohmann/json.hpp"
#include <fstream>
#include <unordered_set>

using json = nlohmann::json;

struct WordPath{ // old thing I was testing ..
    std::string word;
    std::vector<std::string> path;

    [[nodiscard]] std::string printPath() const {
        std::stringstream stream;
        auto it = path.begin();
        while (it != path.end()) {
            stream << *it;
            it++;
            if (it != path.end()) {
                stream << " -> ";
            }
        }

        return stream.str();
    }
};

struct Node {
    Node* parent = nullptr;
    std::vector<Node*> children{};
    std::string TYPE = "";
    std::string Value = "";

    ~Node() {
        for (auto* i : children) {
            delete i;
        }
    }
};

json to_json(const Node* node) {
    if (!node) {
        return nullptr;
    }

    json j;
    j["type"] = node->TYPE;

    // Recursively convert children
    json children_array = json::array();
    for (const auto* child : node->children) {
        children_array.push_back(to_json(child));
    }
    j["children"] = children_array;
    j["value"] = node->Value;

    return j;
}


static std::string eat_word(const std::string& src, size_t& start_pos) { // move until the next space
    std::stringstream stream;
    while (src[start_pos++] != ' ') {
        stream << src[start_pos - 1];
    }
    return stream.str();
}


static Node* parseTree(const std::string& topStr) {
    size_t pos = 0;
    Node* currentNode = nullptr;
    Node* result = nullptr;

    while (pos < topStr.length()) {
        if (topStr[pos] == '(') {
            pos++;
            if (currentNode == nullptr) {
                currentNode = new Node();
                currentNode->TYPE = eat_word(topStr, pos);
            } else {
                auto node = new Node();
                node->TYPE = eat_word(topStr, pos);
                currentNode->children.push_back(node);
                node->parent = currentNode;
                currentNode = node;
            }
            continue;
        }

        if (topStr[pos] == ')') {
            // end of path
            if (currentNode->parent == nullptr) {
                result = currentNode; // the root node (keep it we wanna return that)
            }
            currentNode = currentNode->parent;
            pos++;
            continue;
        }

        if (topStr[pos] == ' ') {
            pos++;
            continue;
        }

        auto node = new Node();
        node->TYPE = "TEXT";
        node->Value = eat_word(topStr, pos);
        node->parent = currentNode;
        currentNode->children.push_back(node);
    }

    return result;
}

void printTree(Node* node, int depth = 0) {
    if (!node) return;

    // Print indentation
    std::string indent(depth * 2, ' ');
    std::cout << indent;

    // Print node information
    std::cout << "Type: " << node->TYPE << ", Value: " << node->Value;

    // Print parent and children info
    std::cout << " Parent: " << (node->parent ? node->parent->TYPE : "NULL");
    std::cout << " Children: " << node->children.size();
    std::cout << std::endl;

    // Recursively print children
    for (Node* child : node->children) {
        printTree(child, depth + 1);
    }
}

static std::vector<std::string> extractTokens(const Node* node) { // given a node, get all the tokens values in it . in the correct order
    std::vector<std::string> result;
    for (auto child : node->children) {
        if (child->TYPE == "TEXT") {
            result.push_back(child->Value);
            continue;
        }
        for (const auto& token: extractTokens(child)) {
            result.push_back(token);
        }
    }
    return result;
}

static Node* flattenTree(Node* node, int depth = 1) { // given a multi level tree, we want to flatten it, moving lower tokens up the tree
    if (depth == 1) {
        Node* c = new Node();
        c->TYPE = node->TYPE;
        c->Value = node->Value;
        for (auto child : node->children) {
            auto tokens = extractTokens(child);

            if (child->TYPE == "TEXT") {
                c->children.push_back(new Node{
                        .parent = c,
                        .children = {},
                        .TYPE = child->TYPE,
                        .Value = child->Value,
                });
            }

            for (const auto& token : tokens) {
                c->children.push_back(new Node{
                        .parent = c,
                        .children = {},
                        .TYPE = child->TYPE,
                        .Value = token,
                });
            }
        }
        return c;
    } else {
        Node* c = new Node();
        c->TYPE = node->TYPE;
        for (auto* child : node->children) {
            auto obj_c = flattenTree(child, depth - 1);
            obj_c->parent = c;
            c->children.push_back(obj_c);
        }
        return c;
    }
}

static std::vector<std::string> splitString(const std::string& str) { // split on spaces
    std::vector<std::string> words;
    std::istringstream iss(str);
    std::string word;
    while (iss >> word) {
        words.push_back(word);
    }
    return words;
}

static std::vector<std::pair<std::string, std::string>> wordTypePair(Node* node) { // dfs get tokens with types
    std::vector<std::pair<std::string, std::string>> pairs;

    for (auto child : node->children) {
        pairs.emplace_back(child->TYPE, child->Value);
        auto _childTokens = wordTypePair(child);
        for (const auto& pair: _childTokens) {
            pairs.emplace_back(pair);
        }
    }

    return pairs;
}

static bool collapseTree(Node* node, std::string type) { // collapse a tag on its children (ie: collapse NOT to get NOT_TOPPING, NOT_STYLE, etc.)
    if (node == nullptr) return false;

    auto it = node->children.begin();
    while (it != node->children.end()) {
        if (collapseTree(*it, type)) {
            // this vector was changed
            return collapseTree(node, type);
        }
        it++;
    }

    if (node->TYPE == type) {
        if (node->parent == nullptr) { // should never happen
            throw "Cannot collapse a root node";
        }


        auto p = node->parent;

        for (auto* c : node->children) {
            c->TYPE = node->TYPE + "_" + c->TYPE;
            c->parent = p;
            p->children.push_back(c);
        }

        p->children.erase(std::find(p->children.begin(), p->children.end(), node));
        node->children.clear();

        node->parent = nullptr;
        delete node;
        return true;
    }

    return false;
}

typedef std::pair<std::string, std::string> LabeledWord;

json processInput(const std::string& jsonStr, const std::string& prefix) { // given an input json build tree and extract useful sequences
    json j = json::parse(jsonStr);

    // Get source and TOP strings
    std::string src = j[prefix + ".SRC"];
    std::string top = j[prefix + ".TOP"];

    json result{};

    auto tree = parseTree(top);
    collapseTree(tree, "NOT");

    result["tree"] = to_json(tree);
    result["src"] = src;
    result["top"] = top;

    // ORDER -> PIZZA , DRINK , NONE
    // PIZZA ORDER -> NUMBER, TOPPING, NOT_TOPPING, COMPLEX_TOPPING, SIZE, STYLE, NONE
    // COMPLEX_TOPPING -> NUMBER, TOPPING, NOT_TOPPING, QUANTITY, NONE (mini-pizza model in some sense)
    // DRINK ORDER -> VOLUME, DRINK_TYPE, CONTAINER_TYPE, NONE
    //

    // find orders boundary
    std::vector<LabeledWord> orders_labels;
    for (const auto* order_node : tree->children) {
        if ((
                order_node->TYPE.find("PIZZAORDER") != std::string::npos ||
                order_node->TYPE.find("DRINKORDER") != std::string::npos
                )) {
            auto words = extractTokens(order_node);
            auto it = words.begin();
            while (it != words.end()) {
                std::string label = order_node->TYPE;
                if (it == words.begin()) {
                    label += "_S";
                }
                orders_labels.emplace_back(label, *it);
                it++;
            }
        } else {
            // misc word
            assert (order_node->TYPE == "TEXT"); // how tf is it not a text ?
            orders_labels.emplace_back("NONE", order_node->Value);
        }
    }
    result["order"] = orders_labels;

    // for each order, find the flattened tokens
    std::vector<std::vector<LabeledWord>> pizza_orders;
    std::vector<std::vector<LabeledWord>> drink_orders;

    std::vector<Node*> pizzas;
    for (auto* order_node : tree->children) {
        if ((
                order_node->TYPE.find("PIZZAORDER") != std::string::npos ||
                order_node->TYPE.find("DRINKORDER") != std::string::npos
        )) {
            auto flattened_order = flattenTree(order_node, 1);
            auto words = wordTypePair(flattened_order);
            delete flattened_order;
            auto it = words.begin();
            std::vector<LabeledWord> order;
            std::string last = "____";
            while (it != words.end()) {
                std::string label = it->first;
                if (it->first != last) {
                    label += "_S";
                    last = it->first;
                }

                if (it->first.find("PIZZAORDER") != std::string::npos ||
                    it->first.find("DRINKORDER") != std::string::npos || it->first == "TEXT") {
                    label = "NONE";
                }

                order.emplace_back(label, it->second);
                it++;
            }

            if (order_node->TYPE.find("PIZZAORDER") != std::string::npos) {
                pizza_orders.push_back(order);
                pizzas.push_back(order_node);
            } else {
                drink_orders.push_back(order);
            }
        }
    }

    result["pizza_orders"] = pizza_orders;
    result["drink_orders"] = drink_orders;

    // now do complex toppings
    std::vector<Node*> expanded_pizzas;
    for (auto* node : pizzas) {
        for (auto n : node->children) {
            expanded_pizzas.push_back(n);
        }
    }
    std::vector<std::vector<LabeledWord>> complex_toppings;
    for (auto* order_node : expanded_pizzas) {
        if ((
                order_node->TYPE.find("COMPLEX_TOPPING") != std::string::npos
        )) {
            auto flattened_order = flattenTree(order_node, 1);
            auto words = wordTypePair(flattened_order);
            delete flattened_order;
            auto it = words.begin();
            std::vector<LabeledWord> order;
            std::string last = "____";
            while (it != words.end()) {
                std::string label = it->first;
                if (it->first != last) {
                    label += "_S";
                    last = it->first;
                }

                if (it->first.find("COMPLEX_TOPPING") != std::string::npos || it->first == "TEXT") {
                    label = "NONE";
                }

                order.emplace_back(label, it->second);
                it++;
            }

            complex_toppings.push_back(order);
        }
    }

    result["complex_toppings"] = complex_toppings;

    delete tree;
    return result;
}

// write a test case to file
static inline void writeTestCase(
        std::unordered_set<std::string>& vocab,
        std::unordered_set<std::string>& tags,
        std::ofstream& words_file,
        std::ofstream& labels_file,
        json& word_labels_pairs
        ) {
    std::string tc = "";
    std::string lc = "";

    for (auto wlp : word_labels_pairs) {
        std::string word = wlp[1];
        std::string tag  = wlp[0];
        tc += word + " ";
        lc += tag + " ";
        vocab.insert(word);
        tags.insert(tag);
    }

    if (!tc.empty()) {
        tc.pop_back();
        lc.pop_back();
        words_file << tc << "\n";
        labels_file << lc << "\n";
    }
}

// write s a hash_set to a file
static inline void writeSet(
        const std::string& path,
        const std::unordered_set<std::string>& set
        ) {
    std::ofstream file(path);
    auto it = set.begin();
    while (it != set.end()) {
        file << *it;
        it++;
        if (it != set.end()) {
            file << "\n";
        }
    }
    file.close();
}

int main() {

//    std::string sample_input = "{\"train.SRC\": \"four party sized pizzas with not much banana peppers and a sprite\", \"train.EXR\": \"(ORDER (DRINKORDER (NUMBER 1 ) (DRINKTYPE SPRITE ) ) (PIZZAORDER (NUMBER 4 ) (SIZE PARTY_SIZE ) (COMPLEX_TOPPING (QUANTITY LIGHT ) (TOPPING BANANA_PEPPERS ) ) ) )\", \"train.TOP\": \"(ORDER (PIZZAORDER (NUMBER four ) (SIZE party sized ) pizzas with (COMPLEX_TOPPING (QUANTITY not much ) (TOPPING banana peppers ) ) ) and (DRINKORDER (NUMBER a ) (DRINKTYPE sprite ) ) )\", \"train.TOP-DECOUPLED\": \"(ORDER (PIZZAORDER (NUMBER four ) (SIZE party sized ) (COMPLEX_TOPPING (QUANTITY not much ) (TOPPING banana peppers ) ) ) (DRINKORDER (NUMBER a ) (DRINKTYPE sprite ) ) )\"}";
//    std::cout << processInput(sample_input, "train").dump(2) << std::endl;
//    // just some testing ..
//    if (true) {
//        return 0;
//    }

    std::string input_file = "../../PIZZA_train.json";
    std::string prefix      = "train";
    std::string output_file = "../../processed_input/train_";

    std::ifstream in_file(input_file);
    std::string line;

    std::ofstream order_out_file(output_file + "orders.txt");
    std::ofstream order_labels_file(output_file + "orders_labels.txt");
    std::unordered_set<std::string> order_tags;

    std::ofstream pizza_orders_out_file(output_file + "pizza_orders.txt");
    std::ofstream pizza_orders_labels_file(output_file + "pizza_orders_labels.txt");
    std::unordered_set<std::string> pizza_orders_tags;

    std::ofstream drink_orders_out_file(output_file + "drink_orders.txt");
    std::ofstream drink_orders_labels_file(output_file + "drink_orders_labels.txt");
    std::unordered_set<std::string> drink_orders_tags;


    std::ofstream complex_topping_out_file(output_file + "complex_topping.txt");
    std::ofstream complex_topping_labels_file(output_file + "complex_topping_labels.txt");
    std::unordered_set<std::string> complex_topping_tags;


    std::unordered_set<std::string> vocab;

    int count = 0;
    while (std::getline(in_file, line)) {
        json j = processInput(line, prefix);

        writeTestCase(vocab, order_tags, order_out_file, order_labels_file, j["order"]);
        for (auto order : j["pizza_orders"]) {
            writeTestCase(vocab, pizza_orders_tags, pizza_orders_out_file, pizza_orders_labels_file, order);
        }
        for (auto order : j["drink_orders"]) {
            writeTestCase(vocab, drink_orders_tags, drink_orders_out_file, drink_orders_labels_file, order);
        }
        for (auto topping : j["complex_toppings"]) {
            writeTestCase(vocab, complex_topping_tags, complex_topping_out_file, complex_topping_labels_file, topping);
        }
        count ++;
        std::cout << "Line: " << count << std::endl;
    }


    in_file.close();
    order_out_file.close();
    order_labels_file.close();
    pizza_orders_out_file.close();
    pizza_orders_labels_file.close();
    drink_orders_out_file.close();
    drink_orders_labels_file.close();
    complex_topping_out_file.close();
    complex_topping_labels_file.close();

    writeSet(output_file + "orders_tags.txt" , order_tags);
    writeSet(output_file + "pizza_orders_tags.txt" , pizza_orders_tags);
    writeSet(output_file + "drink_orders_tags.txt" , drink_orders_tags);
    writeSet(output_file + "complex_topping_tags.txt" , complex_topping_tags);
    writeSet(output_file + "vocab.txt" , vocab);
}