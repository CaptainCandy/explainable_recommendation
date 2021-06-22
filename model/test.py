import wordcloud

# s = "Seen through the eyes of a squad of American soldiers, the story begins with World War II's historic Omaha Beach D-Day invasion, then moves beyond the beach as the men embark on a dangerous special mission. Captain John Miller (Hanks) must take his men behind enemy lines to find Private James Ryan, whose three brothers have been killed in combat. Face with impossible odds, the men question their orders. Why are eight men risking their lives to save just one? Surrounded by the brutal realities of war, each man searches for his own answer -- the strength to triumph over an uncertain future with honor, decency and courage."
# r = "12 Angry Men, by Sidney Lumet, may be the most radical big-screen courtroom drama in cinema history. A behind-closed-doors look at the American legal system as riveting as it is spare, the iconic adaptation of Reginald Rose’s teleplay stars Henry Fonda as the initially dissenting member of a jury of white men ready to pass judgment on a Puerto Rican teenager charged with murdering his father. What results is a saga of epic proportions that plays out in real time over ninety minutes in one sweltering room. Lumet’s electrifying snapshot of 1950s America on the verge of change is one of the great feature-film debuts."
# r = "Eleven jurors are convinced that the defendant is guilty of murder. The twelfth has no doubt of his innocence. How can this one man steer the others toward the same conclusion? It's a case of seemingly overwhelming evidence against a teenager accused of killing his father in 'one of the best pictures ever made'"
# t = "The next instalment in the 'Star Wars' franchise. Rebel Luke Skywalker (Mark Hamill) and his friends continue to battle evil in the form of the decadent galactic empire, headed by Jedi-gone-bad Darth Vader (Dave Prowse, with the voice of James Earl Jones), as the ruthless Palpatine (Ian McDiarmid) sets plans in motion to build a second Death Star with the purpose of destroying the Rebel Alliance."
#
# w = wordcloud.WordCloud(width=800, height=400, background_color='white')
# w.generate(s)
# w.to_file("../example/ryan.jpg")
# w.generate(r)
# w.to_file("../example/angry.jpg")
# w.generate(t)
# w.to_file("../example/sw.jpg")

#%%
import dill as pickle
import pandas as pd
import re


def clean_str(string):
    """
    Tokenization/string cleaning for all datasets except for SST.
    Original taken from https://github.com/yoonkim/CNN_sentence/blob/master/process_data.py
    """
    string = re.sub(r"[^A-Za-z]", " ", string)
    string = re.sub(r"\'s", " \'s", string)
    string = re.sub(r"\'ve", " \'ve", string)
    string = re.sub(r"n\'t", " n\'t", string)
    string = re.sub(r"\'re", " \'re", string)
    string = re.sub(r"\'d", " \'d", string)
    string = re.sub(r"\'ll", " \'ll", string)
    string = re.sub(r",", " , ", string)
    string = re.sub(r"!", " ! ", string)
    string = re.sub(r"\(", " \( ", string)
    string = re.sub(r"\)", " \) ", string)
    string = re.sub(r"\?", " \? ", string)
    string = re.sub(r"\s{2,}", " ", string)
    return string.strip().lower()


dataset_name = "movies"
# reviews_all = []
reviews_all_len = []
with open("../data2014/%s_bert/reviews_all" % dataset_name, "rb") as f:
    while True:
        try:
            line = pickle.load(f)
            line = clean_str(line)
            line = line.split(" ")
            # reviews_all.append(line)
            reviews_all_len.append(len(line))
        except EOFError:
            break
r = pd.Series(reviews_all_len)
print(r.describe())

count = 0
with open("../data2014/%s/vocabulary_all.txt" % dataset_name, "r") as f:
    for line in f:
        count += 1
print(count)