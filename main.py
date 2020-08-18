from glove import load_embedding, tokenize, HashTagMode

# Tokenize tweets (thanks to @tokestermw and @ppope for their basis implementation; see method docs)
tokens = tokenize(
    "Hello #World, this is a sample util from @dhartung to " +
    "illustrate the power of https://github.com/dhartung/python-glove-loader :)",
    hashtag_mode=HashTagMode.REPLACE
)
print(tokens)

# Load whole embedding in memory
glove = load_embedding("./glove.twitter.27B.25d.txt", keep_in_memory=True)
# OR Don't load full embeddings into memory (saves a lot of RAM with large embeddings)
glove = load_embedding("./glove.twitter.27B.25d.txt", keep_in_memory=False)

# Read embeddings
print(glove.get_embedding("potato"))
print(glove.get_embeddings(["potato", "house", "mouse"]))

# Parse tweets
print(
    glove.get_tweet_embeddings(
        "Hello #World, this is a sample util from @dhartung to " +
        "illustrate the power of https://github.com/dhartung/python-glove-loader :)"
    )
)