from transformers import AutoTokenizer
import regex as re
import emoji
import keras
import pickle

class Token:
    def __init__(self, path1,max_len,path2=None):
        """
        :param path1: sub-word tokenizer path
        :param path2: keras tokenizer path
        :param max_len: max length of sequence
        """
        self.tokenizer = AutoTokenizer.from_pretrained(path1)
        if path2 != None:
            with open(path2, "rb") as f:
                self.k_tokenizer = pickle.load(f)
        else:
            self.k_tokenizer = None
        self.max_len = max_len
        emojis = ''.join(emoji.UNICODE_EMOJI.keys())
        self.pattern = re.compile(f'[^ .,?!/@$%~％·∼()\x00-\x7Fㄱ-힣{emojis}]+')
        self.url_pattern = re.compile(
            r'https?:\/\/(www\.)?[-a-zA-Z0-9@:%._\+~#=]{1,256}\.[a-zA-Z0-9()]{1,6}\b([-a-zA-Z0-9()@:%_\+.~#?&//=]*)')

        self.doublespace_pattern = re.compile('\s+')
        self.repeatchars_pattern1 = re.compile('(\w)\\1{2,}')
        self.repeatchars_pattern2 = re.compile('(\W)\\1{2,}')


    def repeat_normalize(self, sent, num_repeats=2):
        if num_repeats > 0:
            sent = self.repeatchars_pattern1.sub('\\1' * num_repeats, sent)
            sent = self.repeatchars_pattern2.sub('\\1' * num_repeats, sent)
            sent = self.doublespace_pattern.sub(' ', sent)
        return sent.strip()

    # clean text
    def clean(self, text):
        text = str(text)
        text = self.pattern.sub(' ', text)
        text = self.url_pattern.sub('', text)
        text = text.strip()
        text = self.repeat_normalize(text, num_repeats=2)
        return text

    # ids to convert token
    def ids_to_token(self, token_lst):
        result = []
        for i in token_lst[1:]:
            if i == 0:
                break
            result.append(self.tokenizer.convert_ids_to_tokens(i))
        return result
    # make token from AutoTokenizer
    def make_token_ori(self, text):
        token_lst = self.tokenizer.encode(
            self.clean(text),
            padding='max_length',
            max_length=self.max_len,
            truncation=True)
        token_lst = self.ids_to_token(token_lst)[:-1]
        return token_lst

    def make_token(self, text):
        token_lst = self.tokenizer.encode(
            self.clean(text),
            padding='max_length',
            max_length=self.max_len,
            truncation=True)
        token_lst = self.ids_to_token(token_lst)[:-1]
        if self.k_tokenizer != None:
            seq = self.k_tokenizer.texts_to_sequences([token_lst])
            token_lst = keras.preprocessing.sequence.pad_sequences(seq, maxlen=self.max_len)
        return token_lst