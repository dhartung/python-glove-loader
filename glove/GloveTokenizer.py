import re
from enum import Enum


class HashTagMode(Enum):
    KEEP = 0
    REPLACE = 1
    ATOMIC = 2
    SPLIT = 3
    SPLIT_USING_PREFIX = 4


def tokenize(text, hashtag_mode: HashTagMode = HashTagMode.SPLIT, use_repeat_token=True, use_elong_token=True):
    """
    Tokenize a text and prepare it for the glove twitter embeddings
    Original code taken from https://gist.github.com/ppope/0ff9fa359fb850ecf74d061f3072633a

    Parameters:
        text: The input text

        hashtag_mode (default: HashTagMode.SPLIT): Defines how to handle hashtags
            HashTagMode.Keep will keep the hashtags unchanged
                (yield ["#helloWorld"])
            HashTagMode.REPLACE will replace the hashtags by a default token
                (yield ["<hashtag>"])
            HashTagMode.ATOMIC will replace the "#" character by the hashtag token
                (yield ["<hashtag>", "helloWorld"])
            HashTagMode.SPLIT is the same as ATOMIC but will split the hashtag in single words
                (yield ["<hashtag>", "hello", "World"])
            HashTagMode.SPLIT_USING_PREFIX is the same as SPLIT but will prefix every output with the hashtag symbol
                (yield ["<hashtag>", "hello", "<hashtag>", "World"])

        use_repeat_token (default: True): Repetitions of punctuation characters (.!?) will be replaced by a token
            (e.g. "hello!!!!" will be converted to ["hello" "!" "<repeat>"])

        use_elong_token: (default: True) Repeats of the last letters of a word are replaced by a special token.
            (e.g. "helloooooooo" will be converted to ["hello", "<elong>"])

        Returns:
            Array of string with tokenized words
    """
    # Different regex parts for smiley faces
    eyes = r"[8:=;]"
    nose = r"['`\-]?"

    regex_flags = re.MULTILINE | re.DOTALL

    # function so code less repetitive
    def re_sub(pattern, repl):
        return re.sub(pattern, repl, text, flags=regex_flags)

    def hashtag(hashtag_text):
        hashtag_text = hashtag_text.group()
        hashtag_body = hashtag_text[1:]
        if hashtag_mode == HashTagMode.KEEP:
            return hashtag_text
        elif hashtag_mode == HashTagMode.REPLACE:
            return "<hashtag>"
        elif hashtag_mode == HashTagMode.ATOMIC:
            return "<hashtag> {}".format(hashtag_body)
        elif hashtag_mode == HashTagMode.SPLIT:
            if hashtag_body.isupper():
                return "<hashtag> {}".format(hashtag_body)
            else:
                return " ".join(["<hashtag>"] + re.findall(r"([a-zA-Z][^A-Z]+|[A-Z]+)", hashtag_body))
        elif hashtag_mode == HashTagMode.SPLIT_USING_PREFIX:
            if hashtag_body.isupper():
                return "<hashtag> {}".format(hashtag_body)
            else:
                return " ".join(
                    ["<hashtag> " + result for result in re.findall(r"([a-zA-Z][^A-Z]+|[A-Z]+)", hashtag_body)])
        else:
            return hashtag_text

    text = re_sub(r"https?:\/\/\S+\b|www\.(\w+\.)+\S*", "<url>")
    text = re_sub(r"@\w+", "<user>")
    text = re_sub(r"{}{}[)dD]+|[)dD]+{}{}".format(eyes, nose, nose, eyes), "<smile>")
    text = re_sub(r"{}{}pP+".format(eyes, nose), "<lolface>")
    text = re_sub(r"{}{}\(+|\)+{}{}".format(eyes, nose, nose, eyes), "<sadface>")
    text = re_sub(r"{}{}[\/|l*]".format(eyes, nose), "<neutralface>")
    text = re_sub(r"/", " / ")
    text = re_sub(r"<3", "<heart>")
    text = re_sub(r"[-+]?[.\d]*[\d]+[:,.\d]*", "<number>")
    text = re_sub(r"#\S+", hashtag)

    if use_repeat_token:
        text = re_sub(r"([!?.]){2,}", r"\1 <repeat>")

    if use_elong_token:
        text = re_sub(r"\b(\S*?)(.)\2{2,}\b", r"\1\2 <elong>")

    # -- I just don't understand why the Ruby script adds <allcaps> to everything so I limited the selection.
    # text = re_sub(r"([^a-z0-9()<>'`\-]){2,}", allcaps)
    text = re_sub(r"([A-Z]{2,})", r"\1 <allcaps>")

    return re.findall(r"([\w<>#]+|[^\s\w])", text.lower(), flags=regex_flags)
