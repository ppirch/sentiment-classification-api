
import re
import torch
import pickle
import string
import numpy as np
from torch import nn
from torch.nn.utils.rnn import pack_padded_sequence
from pythainlp import word_tokenize
from pythainlp.util import emoji_to_thai

np.warnings.filterwarnings('ignore', category=np.VisibleDeprecationWarning)

class BILSTM_Model(torch.nn.Module) :
    def __init__(self, vocab_size, embedding_dim, hidden_dim) :
        super().__init__()
        self.hidden_dim = hidden_dim
        self.dropout = nn.Dropout(0.3)
        self.embeddings = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)
        self.lstm = nn.LSTM(embedding_dim, hidden_dim, batch_first=True)
        self.linear = nn.Linear(hidden_dim, 4)
        
    def forward(self, x, s):
        x = self.embeddings(x)
        x = self.dropout(x)
        x_pack = pack_padded_sequence(x, s, batch_first=True, enforce_sorted=False)
        out_pack, (ht, ct) = self.lstm(x_pack)
        return self.linear(ht[-1])

def preprocess(text):
  def replace_rep(text):
    def _replace_rep(m):
      c,cc = m.groups()
      return f'{c}xxrep'
    re_rep = re.compile(r'(\S)(\1{2,})')
    return re_rep.sub(_replace_rep, text)

  def replace_url(text):
    URL_PATTERN = r"""(?i)\b((?:https?:(?:/{1,3}|[a-z0-9%])|[a-z0-9.\-]+[.](?:com|net|org|edu|gov|mil|aero|asia|biz|cat|coop|info|int|jobs|mobi|museum|name|post|pro|tel|travel|xxx|ac|ad|ae|af|ag|ai|al|am|an|ao|aq|ar|as|at|au|aw|ax|az|ba|bb|bd|be|bf|bg|bh|bi|bj|bm|bn|bo|br|bs|bt|bv|bw|by|bz|ca|cc|cd|cf|cg|ch|ci|ck|cl|cm|cn|co|cr|cs|cu|cv|cx|cy|cz|dd|de|dj|dk|dm|do|dz|ec|ee|eg|eh|er|es|et|eu|fi|fj|fk|fm|fo|fr|ga|gb|gd|ge|gf|gg|gh|gi|gl|gm|gn|gp|gq|gr|gs|gt|gu|gw|gy|hk|hm|hn|hr|ht|hu|id|ie|il|im|in|io|iq|ir|is|it|je|jm|jo|jp|ke|kg|kh|ki|km|kn|kp|kr|kw|ky|kz|la|lb|lc|li|lk|lr|ls|lt|lu|lv|ly|ma|mc|md|me|mg|mh|mk|ml|mm|mn|mo|mp|mq|mr|ms|mt|mu|mv|mw|mx|my|mz|na|nc|ne|nf|ng|ni|nl|no|np|nr|nu|nz|om|pa|pe|pf|pg|ph|pk|pl|pm|pn|pr|ps|pt|pw|py|qa|re|ro|rs|ru|rw|sa|sb|sc|sd|se|sg|sh|si|sj|Ja|sk|sl|sm|sn|so|sr|ss|st|su|sv|sx|sy|sz|tc|td|tf|tg|th|tj|tk|tl|tm|tn|to|tp|tr|tt|tv|tw|tz|ua|ug|uk|us|uy|uz|va|vc|ve|vg|vi|vn|vu|wf|ws|ye|yt|yu|za|zm|zw)/)(?:[^\s()<>{}\[\]]+|\([^\s()]*?\([^\s()]+\)[^\s()]*?\)|\([^\s]+?\))+(?:\([^\s()]*?\([^\s()]+\)[^\s()]*?\)|\([^\s]+?\)|[^\s`!()\[\]{};:'".,<>?«»“”‘’])|(?:(?<!@)[a-z0-9]+(?:[.\-][a-z0-9]+)*[.](?:com|net|org|edu|gov|mil|aero|asia|biz|cat|coop|info|int|jobs|mobi|museum|name|post|pro|tel|travel|xxx|ac|ad|ae|af|ag|ai|al|am|an|ao|aq|ar|as|at|au|aw|ax|az|ba|bb|bd|be|bf|bg|bh|bi|bj|bm|bn|bo|br|bs|bt|bv|bw|by|bz|ca|cc|cd|cf|cg|ch|ci|ck|cl|cm|cn|co|cr|cs|cu|cv|cx|cy|cz|dd|de|dj|dk|dm|do|dz|ec|ee|eg|eh|er|es|et|eu|fi|fj|fk|fm|fo|fr|ga|gb|gd|ge|gf|gg|gh|gi|gl|gm|gn|gp|gq|gr|gs|gt|gu|gw|gy|hk|hm|hn|hr|ht|hu|id|ie|il|im|in|io|iq|ir|is|it|je|jm|jo|jp|ke|kg|kh|ki|km|kn|kp|kr|kw|ky|kz|la|lb|lc|li|lk|lr|ls|lt|lu|lv|ly|ma|mc|md|me|mg|mh|mk|ml|mm|mn|mo|mp|mq|mr|ms|mt|mu|mv|mw|mx|my|mz|na|nc|ne|nf|ng|ni|nl|no|np|nr|nu|nz|om|pa|pe|pf|pg|ph|pk|pl|pm|pn|pr|ps|pt|pw|py|qa|re|ro|rs|ru|rw|sa|sb|sc|sd|se|sg|sh|si|sj|Ja|sk|sl|sm|sn|so|sr|ss|st|su|sv|sx|sy|sz|tc|td|tf|tg|th|tj|tk|tl|tm|tn|to|tp|tr|tt|tv|tw|tz|ua|ug|uk|us|uy|uz|va|vc|ve|vg|vi|vn|vu|wf|ws|ye|yt|yu|za|zm|zw)\b/?(?!@)))"""
    return re.sub(URL_PATTERN, 'xxurl', text)

  def replace_emoji(text):
    return emoji_to_thai(text, delimiters=(' ', ' '))

  def replace_punctuation(text):
    punctuation = string.punctuation.replace(".", "") # remove all punctuation except . because we want to keep infomation some word such as "อ.", "จ." and etc.
    punctuation_translator = str.maketrans('', '', punctuation)
    return text.translate(punctuation_translator)

  preprocess_text = text.lower().strip()
  preprocess_text = replace_url(preprocess_text)
  preprocess_text = replace_rep(preprocess_text)
  preprocess_text = replace_emoji(preprocess_text)
  preprocess_text = replace_punctuation(preprocess_text)
  return preprocess_text

class MySentimentModel:

  def __init__(self) -> None:
      with open("./dumps/vocab2index.pkl", "rb") as f:
          self.vocab2index = pickle.load(f)
      with open("./dumps/id2label.pkl", "rb") as f:
          self.id2label = pickle.load(f)
      self.model = BILSTM_Model(len(self.vocab2index), 300, 128)
      self.model.load_state_dict(torch.load('./weights/bilstm_model.pth', map_location=torch.device('cpu')))

  def encode_sentence(self, text, vocab2index, N=70):
      tokenized = word_tokenize(text)
      encoded = np.zeros(N, dtype=int)
      enc1 = np.array([vocab2index.get(word, vocab2index["UNK"]) for word in tokenized])
      length = min(N, len(enc1))
      encoded[:length] = enc1[:length]
      return encoded, length
      
  def predict(self, text) -> str:
      preprocess_text = preprocess(text)
      x, l = self.encode_sentence(preprocess_text,self.vocab2index )
      x, l = torch.Tensor(x).long(), torch.Tensor([l]).long()
      prediction = self.model(x.unsqueeze(0), l)[0].detach().numpy()
      prediction_class = np.argmax(prediction)
      return self.id2label[prediction_class]