import regex as re
import os
import json 


# Dataset Loader
class TextFolder():
    '''
    Helps Us Read all the Text Files
    Input is string which is the folder in which all text files are there.
    '''
    def __init__(self, folder_path:str) -> None:
        self.folder_path = folder_path
        self._GPT4_SPLIT_PATTERN = re.compile(r"""'(?i:[sdmt]|ll|ve|re)|[^\r\n\p{L}\p{N}]?+\p{L}+|\p{N}{1,3}| ?[^\s\p{L}\p{N}]++[\r\n]*|\s*[\r\n]|\s+(?!\S)|\s+""")
    
    # Reads the text Files data
    def read(self):
        '''
        An Iterator which
        Reads text files from the given folder-path.
        Converts the Text Files Data into UTF-8 Encoded strings.
        '''
        for root, _, files in os.walk(self.folder_path):
            for file_name in files:
                full_file_path = os.path.join(root, file_name)

                with open(full_file_path, "r", encoding="utf-8") as f:
                    yield f.read()  

    # Splits the Raw Text into Tokens        
    def _preprocessor(self, raw_text):
        for doc_text in raw_text:
            for match in self._GPT4_SPLIT_PATTERN.finditer(doc_text):
                yield match.group(0)       


class BPE():
    '''
    Main Byte-Pair Encoding Class
    Requires folder_path 
    vocab_size and n_merges required only when training
    '''
    
    def __init__(self, vocab_size=30000, n_merges=1000) -> None:
        self._GPT4_SPLIT_PATTERN = re.compile(r"""'(?i:[sdmt]|ll|ve|re)|[^\r\n\p{L}\p{N}]?+\p{L}+|\p{N}{1,3}| ?[^\s\p{L}\p{N}]++[\r\n]*|\s*[\r\n]|\s+(?!\S)|\s+""") 
        self.vocab_size = vocab_size
        self.n_merges = n_merges
        
        
    def _get_pair_counts(self, corpus:dict)->dict:
        pair_counts = {}

        for word, freq in corpus.items():
            for i in range(len(word) - 1):
                pair = (word[i], word[i+1])

                if pair in pair_counts:
                    pair_counts[pair] += freq
                else:
                    pair_counts[pair] = freq

        return pair_counts


    def _merge_corpus(self, corpus:dict, pair, idx):
        '''
        Merges 2-tokens into one pair and gives them new idx. 
        ids: The token list encoded in utf
        pair: The pair we want to replace with
        idx: new idx we want to give that the pair
        '''
        new_corpus = {}

        for word, freq in corpus.items():
            new_word = []
            i = 0

            while i < len(word):
                # If we are Not at Last Position ie. len(ids)-1 then Replace if the pair matches
                if i < len(word)-1 and word[i]==pair[0] and word[i+1]==pair[1]:
                    new_word.append(idx)
                    i += 2 # Why +=2? Because Iterating over pair
                else:
                    new_word.append(word[i])
                    i += 1

            new_word = tuple(new_word)

            if new_word in new_corpus:
                new_corpus[new_word] += freq
            else:
                new_corpus[new_word] = freq

        return new_corpus

    def save(self, vocab_path:str, merges_path:str)->None:
        '''
        Saves the Vocabulary and Merges Dictionary
        '''
        self.vocab_path, self.merges_path = vocab_path, merges_path

        serialized_vocab = {str(k): list(v) for k, v in self.vocab.items()} # Serialized The Vocabulary As JSON does support 'bytes' type

        with open(self.vocab_path, "w") as f:
            json.dump(serialized_vocab, f, indent=2)

        with open(self.merges_path, "w") as f:
            json.dump({str(k): v for k, v in self.merges.items()}, f, indent=2)
    
    def load(self, vocab_path:str, merges_path:str):
        '''
        Loads the Vocabulary And Merges Dictionary
        '''
        with open(vocab_path) as f:
            self.vocab = {int(k): bytes(v) for k, v in json.load(f).items()}

        with open(merges_path) as f:
            raw = json.load(f)
            self.merges = {tuple(map(int, k.strip("()").split(","))): v for k, v in raw.items()}
        
        return self.vocab, self.merges

    # Training Functions Creates Vocabulary and Merges Dictionary which is used by Encoder/Decorder Functions
    def train(self, folder_path:str, print_progress:bool=False)->None:
        '''
        Trains On Custom Data.
        Reads the Text Dataset From a Folder
        '''
        self.corpus = {}
        self.cache = {}
        self.merges = {} # {(int, int) : int} # Merge Dictionary
        
        helper_class = TextFolder(folder_path=folder_path)
        self.raw_text = helper_class.read() # Gets the Raw Text from all the textt files 

        # Coverting Each Word into UTF-8 Encoding And Regex Splitting GPT4 Style
        for token in helper_class._preprocessor(self.raw_text):
            if token in self.cache:
                word = self.cache[token]
            else:
                b = token.encode("utf-8")   # avoid double encode
                word = tuple(b)
                self.cache[token] = word

            self.corpus[word] = self.corpus.get(word, 0) + 1


        # Main Training Loop: Merging Frequently occurring Tokens 
        for i in range(self.n_merges):
            count = self._get_pair_counts(self.corpus) # Gets the Count of Each Pair
            pair = max(count, key=count.get) # Gets the Count of the most frequent pair
            idx = (self.vocab_size - self.n_merges) + i # Gives the most occurring pair a new index

            if(i%100==0 and print_progress==True):
                print(f"Merge No. {i:3d} | Merging {pair} as new idx {idx}") 

            self.corpus = self._merge_corpus(self.corpus, pair, idx) # Merges the pair
            self.merges[pair] = idx # Updates the merges Dictionary


        # Creating Vocabulary
        self.vocab = {idx: bytes([idx]) for idx in range(256)} 
        '''
        Why range(256)? Standard for byte-level BPE to handle any Unicode 
        text via UTF-8 bytes—ensures no unknown characters by falling back to individual bytes if needed.
        '''
        for (p0, p1), idx in sorted(self.merges.items(), key=lambda x: x[1]): # key=... gurantees the order
           self.vocab[idx] = self.vocab[p0] + self.vocab[p1] # Byte Addition
        

        # Saving the Vocab & Merges Dict
        self.save("cache/vocab.json", "cache/merges.json")
    

    def _merge_tokens(self, tokens, pair, idx):
        '''
        Merges the given pair into one single Token. 
        Then returns the new_tokens list
        '''
        new_tokens = []
        i = 0

        while i < len(tokens):
            if i < len(tokens) - 1 and tokens[i] == pair[0] and tokens[i+1] == pair[1]:
                new_tokens.append(idx)
                i += 2
            else:
                new_tokens.append(tokens[i])
                i += 1

        return new_tokens

    def _get_pairs(self, ids):
        return [(ids[i], ids[i+1]) for i in range(len(ids)-1)]
    
    # Encoder Function
    def encoder(self, text: str) -> list[int]:
        '''
        Convert str to int
        Convert text to list of tokens
        '''
        
        self.vocab, self.merges = self.load("cache/vocab.json", "cache/merges.json")

        final_ids = []

        for match in self._GPT4_SPLIT_PATTERN.finditer(text): 
            tokens = list(match.group(0).encode("utf-8")) # Text to Regex Split GPT4 Style to UTF-8 Encoding

            while len(tokens) >= 2:
                pairs = self._get_pairs(tokens)

                pair = min(pairs, key=lambda p: self.merges.get(p, float("inf"))) 

                if pair not in self.merges:
                    break # That is if the pair is not found in the merges Dictionary.

                idx = self.merges[pair] # Finds the Id of the Pair in the Merges Dictionary if it is there else the above code breaks the while loop.
                tokens = self._merge_tokens(tokens, pair, idx) # Merges the tokens which appeared in the Merges Dictionary

            final_ids.extend(tokens) # Adds the Merged Tokens encoded tokens into the encoded list.

        return final_ids


    # Decorder Function
    def decode(self, ids, error_state="replace")->str:
        '''
        Given a set of ids, decode into text. 
        Basically int to str
        '''
        tokens = bytearray()
        for idx in ids:
            tokens.extend(self.vocab[idx])
        
        return tokens.decode("utf-8", errors=error_state) # When doing ids =[128], it'll throw an error, thus replacing it with <?>


if __name__ == "__main__":
    import time
    start_time=time.time()
    BPE().train(folder_path="data")
    end_time=time.time() - start_time

    print(f"\nTime Taken To Train: {end_time//60:.0f} minutes")
