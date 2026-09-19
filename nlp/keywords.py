from collections import Counter
from kiwipiepy import Kiwi

kiwi = Kiwi()
STOPWORDS = {
    "민원","관련","요청","부탁","확인","조치","문제","주민",
    "때문","정도","이용","계속","주변","최근"
}

def extract_keywords(texts, top_n=15):
    counter = Counter()
    for text in texts:
        for token in kiwi.tokenize(str(text)):
            if token.tag.startswith(("N", "SL")) and len(token.form) >= 2:
                if token.form not in STOPWORDS:
                    counter[token.form] += 1
    return counter.most_common(top_n)
