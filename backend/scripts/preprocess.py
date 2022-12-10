import pandas as pd
import os
import re

# Define text cleaning pattern
urlPattern        = r"((http://)[^ ]*|(https://)[^ ]*|(www\.)[^ ]*)"
userPattern       = '@[^\s]+'
hashtagPattern    = '#[^\s]+'
alphaPattern      = "[^a-z0-9<>]"
sequencePattern   = r"(.)\1\1+"
seqReplacePattern = r"\1\1"

# Define emojis cleaning pattern
smileemoji        = r"[8:=;]['`\-]?[)d]+"
sademoji          = r"[8:=;]['`\-]?\(+"
neutralemoji      = r"[8:=;]['`\-]?[\/|l*]"
lolemoji          = r"[8:=;]['`\-]?p+"


class Preprocess:
    def __init__(self):
        parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        contraction_path = parent_dir + '/content/dataset/contractions.csv'

        # Load Enlgish contraction dictionary
        contractions = pd.read_csv(contraction_path, index_col='Contraction')
        contractions.index = contractions.index.str.lower()
        contractions.Meaning = contractions.Meaning.str.lower()
        self.contractions_dict = contractions.to_dict()['Meaning']

    def preprocess(self, text):
        # 1, Convert to lower case
        text = text.lower()

        # 2, Replace all URls with '<url>'
        text = re.sub(urlPattern, '<url>', text)
        
        # 3, Replace all @USERNAME to '<user>'.
        text = re.sub(userPattern, '<user>', text)
        
        # 4, Replace 3 or more consecutive letters by 2 letter.
        text = re.sub(sequencePattern, seqReplacePattern, text)

        # 5, Replace all emojis.
        text = re.sub(r'<3', '<heart>', text)
        text = re.sub(smileemoji, '<smile>', text)
        text = re.sub(sademoji, '<sadface>', text)
        text = re.sub(neutralemoji, '<neutralface>', text)
        text = re.sub(lolemoji, '<lolface>', text)

        # 6, Remove Contractions
        for contraction, replacement in self.contractions_dict.items():
            text = text.replace(contraction, replacement)

        # 7, Removing Non-Alphabets and replace them with a space
        text = re.sub(alphaPattern, ' ', text)

        # 8, Adding space on either side of '/' to seperate words.
        text = re.sub(r'/', ' / ', text)
        
        return text
        