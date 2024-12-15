import pandas as pd
import numpy as np
from num2words import num2words
from nltk.stem import SnowballStemmer
from nltk.tokenize import TreebankWordTokenizer
from torch.utils.data import Dataset, DataLoader
import re
import torch
import torch.nn as nn
from sklearn.preprocessing import OneHotEncoder
import pickle

device = torch.device("cuda" if  torch.cuda.is_available() else "cpu")

# Read the Dataset given path to JSON file
# input: JSON file   -> output: list of size 4 (sentence, EXR, TOP, TOP_DECOUPLED) * number of strings
# we will use this to read the Training/ evaluation / test datasets. 
def read_dataset(path: str):
    data = pd.read_json(path, lines = True)
    columns = data.columns.tolist()
    parsed_json = [None]*len(columns)
    for i in range(0,len(columns)):
        parsed_json[i] = data[columns[i]] # IDK will it be easier to us to work with pandas or numpy
    return parsed_json # we store data in list of PD.Series for now

# the function takes SERIES of string sentences -> outputs SERIES of String sentences
def pre_text_normalization(sentences : pd.Series, lowerize =-1):
# Words to Lower
    if lowerize == -1:
        sentences = sentences.str.lower()
# SIZES

# after asking the TA, stating one format isn't a good idea so i won't standaradize the format
# so things like Party - size , party size to standaradize this i will just remove the '-'
    sentences = sentences.str.replace(r"-"," ",regex=True)
    
    sentences = sentences.str.replace(r"\s{2}",r" ",regex=True)

# sometimes they refer to pizza as pie : WE ONLY SELL PIZZA
    sentences = sentences.str.replace("pie", "pizza")

# Gluten - free we can leave it like that for now (may standardize it to gluten_free in future)

# Quantities
    '''
    Now i want to take into consider quantities that means less of topping
    something like 
    not much, not many, a little bit of, just a tiny bit of, just a bit, only a little, just a little, a bit, a little
    we will leave those quantities like that for now and see in the future if we will change them
    for quantities that mean much:
    a lot of, lots of
    '''

# there are alot of Quantitative items 3 pies, Three pies ..
# normalize digits to words 
    sentences = sentences.str.replace(r"\b([0-9]+)\b",lambda match: num2words(int(match.group(1))),regex=True)
    
# Negation
    '''
    There is multiple ways of negation, what i found while searching:
    Without, hold the, With no(t), no, avoid, hate
    i want complex words like (hold the , without) to be converted int no
    we won't change those for now because i want to try learn the context of negation
    '''
# TOPPINGS 
# (I think BBQ topping needs to be paired with things, it's always written as bbq_chicken, bbq_sauce, bbq_pulled_pork...)
# i think this is oversimplification and i will let the sequence model decide this
# To be decided later

# DRINKS

    sentences = sentences.str.replace(r"\bmilliliter\b", "ml", regex=True)

    sentences = sentences.str.replace(r"\bfluid\b","fl",regex=True)
# sometimes people say pepsi, sometimes pepsis so i don't want plurals -> let's stem
    sentences = sentences.str.replace(r"\b(\w\w+)e?s\b",r"\1",regex=True)
# sometimes san pellegrino is said pellegrino only
    sentences = sentences.str.replace(r"\bsan\s(pellegrino)\b",r"\1",regex=True)
# sometimes wrote zeros as zeroe
    sentences = sentences.str.replace(r"\b(zero)e\b",r"\1",regex=True)
# sometimes people write iced instead of ice, sized instead of size
    sentences = sentences.str.replace(r"\b(ice|size)d\b",r"\1",regex=True)
# DOCTOR PEPPER convert dr to doctor , peper to pepper
    sentences = sentences.str.replace(r"\bdr\b",r"doctor",regex=True)
    sentences = sentences.str.replace(r"\bpeper\b",r"pepper",regex=True)
    
    return sentences
# Stemmer 
def snow_ball_stemmer(vocab):
    stemmer = SnowballStemmer("english")
    if isinstance(vocab,set):
        vocab = set([stemmer.stem(word) for word in vocab])
        return vocab
    else:
        vocab = vocab.apply(lambda words: [stemmer.stem(word) for word in words])
        return vocab
def prepend_start_marker(lst):
    lst.insert(0, "<S>")
    return lst
def append_end_marker(lst):
    lst.append("</S>")
    return lst
# the function takes SERIES of string sentences -> outputs SET of vocab and , SERIES of list of tokens
def tokenization(sentences: pd.Series, tokenizesentences = -1, no_pad=False,eval=False):
    # merge the whole series int one sentence to make the vocab extracting faster
    all_words = ' '.join(sentences)
    # used penn treebank tokenizer
    tokenizer = TreebankWordTokenizer()
    all_words = tokenizer.tokenize(all_words)
    if tokenizesentences != -1:
        if not eval:
            with open('DRINK_MWE_TOKENS.pkl', 'rb') as file:
                loaded_tokenizer = pickle.load(file)
        else:
            with open("eval_DRINK_MWE_TOKENS.pkl","rb") as file:
                loaded_tokenizer = pickle.load(file)
        print(all_words)
        all_words = loaded_tokenizer.tokenize(all_words)
    # keep the unique 
    vocab = set(all_words)
    
    sentences = sentences.apply(tokenizer.tokenize)
    if tokenizesentences != -1:
        sentences = sentences.apply(loaded_tokenizer.tokenize)
    
    sentences.fillna("",inplace=True)

    # convert tokenized_sentences into padded lists so that they have same dimension
    
    max_length = sentences.map(len).max()
    padded_tokenized_sentences = sentences
    if not no_pad:
        sentences = sentences.apply(lambda x: prepend_start_marker(x))
        sentences = sentences.apply(lambda x: append_end_marker(x))
        padded_tokenized_sentences = sentences.apply(lambda x: x + [np.nan] * (max_length - len(x)))
        padded_tokenized_sentences = pd.DataFrame(padded_tokenized_sentences.tolist())
        padded_tokenized_sentences.fillna(0,inplace = True)
   
   # negation check regex : \b(?<=not?)(.*?)(?=(\.|,|$|and))\b (for the future maybe ?)
    return vocab, padded_tokenized_sentences
def simple_convert_strings_to_regex(mylist: list[str]):
    for i, string in enumerate(mylist):
        string = string.replace(" ", r"\s")
        string = r"\b"+string+r"\b"
        mylist[i] = string
    return mylist

def extract_pizza_drinks(parsed_tree: pd.Series): # the tree is a SERIES of format that is like this (ORDER (DRINK,))....

# i extract PIZZAORDER node if exist, and DRINKORDER node if exist
    
    pizza_orders, drink_orders = None, None
    # pattern that is interested in matching everything between "(ORDER" and ")"
    # deleting parenthesis -> ease next steps of regex

    order_pattern = r"(?<=\(ORDER)(.*)(?=\))"   
    
    # (ORDER i want to eat (PIZZAORDER) without (NOT...)) This regex will extract i want to eat, without
    # anything between ) "" (, anything after (ORDER  -> Extract all Words that may be NONE (needs to be refined)
    extracted_words_before_parsing = r"(?:(?:\(ORDER\s+)|(?:\)))([^()]+)(?=[\s(]+)"
    # extract None words first before removing the (ORDER -> (can be done after removing it also)
    none_words = parsed_tree.str.extractall(extracted_words_before_parsing).iloc[:,0].str.strip()
    # clean the none words extracted
    none_words.dropna()
    none_words.drop_duplicates(inplace=True)

    temp = set()
    for word in none_words:
        x = word.split()
        temp.update(set(x))
    none_words = list(temp)
    del temp
    # use the order_pattern
    extracted_orders = parsed_tree.str.extractall(order_pattern).iloc[:,0].str.strip()

    # i have interested parenthesises (PIZZAORDER)(DRINKORDER)
    # that may be interrupted by some none tokens -> i don't them anymore after i extracted them above
    # so i will delete them from the Strings to make the parser work correctly
    regex_list = simple_convert_strings_to_regex(none_words.copy())

    for word in regex_list:
        extracted_orders = extracted_orders.str.replace(word,"",regex=True)

    
    # every pizzaorder either ends with pizzaorder, drink order, or end of sentence, same for drinks
    pizza_order_pattern = r"(?<=\(PIZZAORDER\s)(.*?)(?=\)\s*$|\)\s*\(DRINKORDER|\)\s*\(PIZZAORDER)"
    drink_order_pattern = r"(?<=\(DRINKORDER\s)(.*?)(?=\)\s*$|\)\s*\(DRINKORDER|\)\s*\(PIZZAORDER)"


    pizza_orders = extracted_orders.str.extractall(pizza_order_pattern)
    
    drink_orders = extracted_orders.str.extractall(drink_order_pattern)

    # remove the sentences where the user didn't order drinks
    drink_orders = drink_orders.dropna().reset_index(drop=True)
    # remove the sentences where the user didn't order pizzas
    pizza_orders = pizza_orders.dropna().reset_index(drop=True)
    
    del extracted_orders

    # return series of pizzaorders (TOPPING)(STYLE....), series of drinkorders of same format
    # series of none_words i, 'd, want, .... 
    return pizza_orders, drink_orders, none_words
# takes a pd.Series of format (TOPPING)(STYLE)...
# returns the words under every label
def extract_nodes(pizza_orders:pd.Series,drink_orders:pd.Series):
    drink_nodes, pizza_nodes = [] ,[]
    if np.any(pizza_orders) :
        pizza_node_attributes = ["NUMBER","SIZE","TOPPING","QUANTITY","STYLE"]
        for attribute in pizza_node_attributes:
            node_pattern = r"(?<=\("+attribute+r")(.*?)(?=\))"
            pizza_order_node = pizza_orders.str.extract(node_pattern)
            pizza_order_node.rename(columns={0:attribute},inplace=True)
            pizza_nodes.append(pizza_order_node)
            
    if np.any(drink_orders) :
        drink_node_attributes = ["NUMBER","SIZE","DRINKTYPE","CONTAINERTYPE","VOLUME"]
        for attribute in drink_node_attributes:
            node_pattern = r"(?<=\("+attribute+r")(.*?)(?=\))"
            drink_order_node = drink_orders.str.extract(node_pattern)
            drink_order_node.rename(columns={0:attribute},inplace=True)
            drink_nodes.append(drink_order_node)
    return pizza_nodes, drink_nodes
def clean_extracted_nodes(pizza_nodes: list[pd.Series], drink_nodes: list[pd.Series]):
    # i want to refine the extracted nodes since the one parsed from previous step has
    # alot of nans so i will drop those, normalize the text and drop the duplicates
    # after this step i can start labling the text
    new_pizza_nodes, new_drink_nodes = [], []
    drink_node_attributes = ["NUMBER","SIZE", "DRINKTYPE","CONTAINERTYPE","VOLUME"]
    pizza_node_attributes = ["NUMBER","SIZE","TOPPING","QUANTITY","STYLE"]
    for i in range(0,len(pizza_nodes)):

        node = pizza_nodes[i].dropna().reset_index(drop=True)
        # so that if a node  wasn't parsed (Technically impossible)

        j = i
        while pizza_node_attributes[j] != node.columns[0]:
            new_pizza_nodes.append(None)
            j +=1

        # convert the node from Dataframe of one series to one series 
        # so that we can use the series.str methods
        node = node.iloc[:,0]
        # if there was duplicates drop it to make normalization faster
        node.drop_duplicates(keep='first',inplace=True)
        
        node = pre_text_normalization(node)
        # after normalization duplicates will appear so delete them
        node.drop_duplicates(keep='first',inplace=True)
        
        node = node.reset_index(drop=True)
        # ensure there is no spaces
        node = node.str.strip()

        new_pizza_nodes.append(node)
    # because if style or the latter nodes  wasn't found  (technically impossiblek)
    while len(new_pizza_nodes) <5:
        new_pizza_nodes.append(None)

    for i in range(0,len(drink_nodes)):
        # same for Drinks
        node = drink_nodes[i].dropna().reset_index(drop=True)
        
        j = i
        while drink_node_attributes[j] != node.columns[0]:
            new_drink_nodes.append(None)
            j +=1
        
        node = node.iloc[:,0]
        
        node.drop_duplicates(keep='first',inplace=True)
        
        node = pre_text_normalization(node)
        
        node.drop_duplicates(keep='first',inplace=True)
        
        node = node.reset_index(drop=True)
        
        node = node.str.strip()
        
        new_drink_nodes.append(node)

    while len(new_drink_nodes) <5:
        new_drink_nodes.append(None)
    
    return new_pizza_nodes, new_drink_nodes
def create_labeled_eval_vocab(convertor):
    eval_vocab = pd.read_csv("eval_vocab.csv")
    with open("eval_DRINK_MWE_TOKENS.pkl","rb") as file:
        eval_mwe = pickle.load(file)
    eval_vocab["0"].apply(eval_mwe.tokenize)
    eval_vocab["1"] = None
    labels = [None, None, None, None, None, None, None, None, None,None,pd.Series("pizza")]

    csv_file_names = ["number", "size", "none","topping","quantity","style","drink_type","container_type","volume","negation"]

    for i, csv in zip(range(0,len(labels)), csv_file_names):
        labels[i] = pd.read_csv(f"./eval_labels/{csv}.csv").iloc[:,0]
        labels[i] = labels[i].str.strip()
    csv_file_names.insert(11,"pizza")
    for i in range(0,11):
        eval_vocab.loc[eval_vocab["0"].isin(labels[i]),"1"] = csv_file_names[i]
    eval_vocab["encoded_tokens"] =  eval_vocab["0"].map(convertor.word2id)
    eval_vocab["encoded_labels"] = eval_vocab["1"].map(convertor.labels2id)

    return eval_vocab
def one_hot_encoding(vocab):
    unlabeled_vocab = vocab.to_numpy().reshape(-1,1)
    
    encoder = OneHotEncoder()
    
    encoder = encoder.fit(unlabeled_vocab)

    return encoder
def create_labeled_vocab(vocab: pd.DataFrame,eval=False):
    if isinstance(vocab, type(None)):
        vocab = pd.read_csv("vocab.csv")
   
    # add unknown for future when testing eval set
   
    vocab.loc[-1] = "UNK"

    vocab = vocab.reset_index(drop=True)

    vocab["1"] = "UNK"
    # because pizza, negation aren't put within () in preprocessing
    
    # i put them by myself to remove them from the None set
    labels = [None, None, None, None, None, None, None, None, None,None,pd.Series("pizza")]

    csv_file_names = ["number", "size", "none","topping","quantity","style","drink_type","container_type","volume","negation"]

    for i, csv in zip(range(0,len(labels)), csv_file_names):
            labels[i] = pd.read_csv(f"./labels/{csv}.csv").iloc[:,0]
            labels[i] = labels[i].str.strip()

    csv_file_names.insert(11,"pizza")
    for i in range(0,11):
        vocab.loc[vocab["0"].isin(labels[i]),"1"] = csv_file_names[i]
 
    # returns vocab against labels
    vocab_encoder = one_hot_encoding(vocab[vocab.columns[0]])

# this will be as used as our target outputs 
    csv_file_names.append("UNK")
    label_encoder = one_hot_encoding(pd.Series(csv_file_names))

    encoded_tokens = vocab_encoder.transform(vocab["0"].to_numpy().reshape(-1,1))
    encoded_labels = label_encoder.transform(vocab["1"].to_numpy().reshape(-1,1))

    vocab.rename(columns={"0": "tokens","1": "labels"},inplace=True)

    vocab["encoded_tokens"] = pd.Series([x.toarray().argmax(axis=1)[0] for x in encoded_tokens])
    
    vocab["encoded_labels"] = pd.Series([x.toarray().argmax(axis=1)[0] for x in encoded_labels])
    
    # write for future purposes instead of going through this loop again
    vocab.to_csv("labeled entities.csv",index=False)

    return vocab, vocab_encoder, label_encoder
class conversions():
    def __init__(self,vocab,label_encoder):
        self.token_to_id = dict(zip(vocab["tokens"], vocab["encoded_tokens"]))
        self.token_to_id["PAD"] = vocab.shape[0]
        self.token_to_id["<S>"] = vocab.shape[0]  + 1
        self.token_to_id["</S>"] = vocab.shape[0] + 2

        self.label_to_id = dict(zip(label_encoder.categories_[0],range(len(label_encoder.categories_[0]))))
        self.label_to_id["<S>"]  = 12
        self.label_to_id["</S>"] = 13

        self.token_to_label = dict(zip(vocab["tokens"], vocab["encoded_labels"]))

        self.token_to_label["<S>"] = 12
        self.token_to_label["</S>"] = 13

        self.id_to_token = dict(zip(vocab["encoded_tokens"],vocab["tokens"]))
        self.id_to_token[vocab.shape[0] + 1] = "<S>"
        self.id_to_token[vocab.shape[0] + 2] = "</S>"
        
        self.id_to_label = dict(zip(vocab["encoded_labels"],vocab["labels"]))
        self.id_to_token[12] = "<S>"
        self.id_to_token[13] = "</S>"

    def word2id(self,word):
        x = self.token_to_id.get(word, None)
        if not isinstance(x,type(None)):
            return x
        else:
            return self.token_to_id.get("UNK",None)
        
    def labels2id(self,word):
        return self.label_to_id.get(word, None)
    
    def word2label(self,word):
        return self.token_to_label.get(word,-1)
    
    def id2token(self,number):
        return self.id_to_token.get(int(number),"UNK")
    
    def id2label(self,number):
        return self.id_to_label.get(number, None)
# we can use the DataSet class from pytorch to facilitate 
# batch divisions and data preparation
class SimpleDataset(Dataset):
    def __init__(self, input_indices, labels):
        self.input_indices = input_indices
        self.labels = labels

    def __len__(self):
        return len(self.input_indices)

    def __getitem__(self, idx):
        return self.input_indices[idx], self.labels[idx]
def to_pass_size_as_arg(size):
    def integer_to_one_hot(index):
        # Create a zero vector of length num_classes
        one_hot = torch.zeros(size).type(torch.float32)
        # Set the position corresponding to the index to 1
        one_hot[int(index)] = 1
        return one_hot
    return integer_to_one_hot
class RNN(nn.Module):
    def __init__(self, input_size, num_classes, hidden_size):
        super(RNN,self).__init__()
        self.embedding = nn.Embedding(num_embeddings=input_size+3,embedding_dim=100,padding_idx=input_size).to(device)
        nn.init.kaiming_uniform_(self.embedding.weight,mode="fan_in", nonlinearity="relu")
        self.hidden_size = hidden_size
        # batch_first = True means that batch is the first dimension
        # shape : batch_first, seq, input_size
        self.lstm = nn.LSTM(input_size=100,hidden_size=hidden_size, batch_first=True, bidirectional=True).to(device)
        self.dropout = nn.Dropout(p=0.5)
        for name, param in self.lstm.named_parameters():
            if 'weight_ih' in name:
                nn.init.kaiming_uniform_(param.data)  # Kaiming initialization for input weights
            elif 'weight_hh' in name:
                nn.init.orthogonal_(param.data)  # Orthogonal initialization for hidden weights
            elif 'bias' in name:
                nn.init.zeros_(param.data)  # Initialize biases to zero
        # linear layer : from hidden RNN to Output
        self.fc = nn.Linear(hidden_size*2, num_classes).to(device)
        nn.init.kaiming_uniform_(self.fc.weight,mode="fan_in", nonlinearity="relu")

    def forward(self, input):
        # input is batch, seq cuz it's integer indices

        embed = self.embedding(input)
        # for LSTM we need initial tensor state + initial hidden state
        # where initial tensor state is called (cell)
        # 1 : num of layers , batch size , hidden_size
        c_0 = torch.zeros(2,input.size(0),self.hidden_size).to(device)
        h_0 = torch.zeros(2,input.size(0),self.hidden_size).to(device)
        # output of self.rnn : out feature, hidden_state(n)
        out, _ = self.lstm(embed,(h_0,c_0))
        # size of output = batch, seq_length, hidden_size
        out = self.fc(out)
    
        return out