"""VerbalDataset"""
from torchtext.data import Field, Example, Dataset
from underthesea import word_tokenize
from tqdm import tqdm_notebook
from utils.constants import (
    ANSWER_TOKEN, ENTITY_TOKEN, SOS_TOKEN, EOS_TOKEN,
    SRC_NAME, TRG_NAME
)

class VerbalDataset(object):
    """VerbalDataset class"""
                                         
    def __init__(self,train,val,test):
        self.train = train
        self.val = val
        self.test = test
        self.train_data = None
        self.valid_data = None
        self.test_data = None
        self.src_field = None
        self.trg_field = None

    def _make_torchtext_dataset(self, data, fields):
        examples = [Example.fromlist(i, fields) for i in tqdm_notebook(data)]
        return Dataset(examples, fields)

    def load_data_and_fields(self, ):
        """
        Load verbalization data
        Create source and target fields
        """
        train, test, val = [], [], []
        
        train = self.train
        val = self.val
        test = self.test

        # create fields
        self.src_field = Field(tokenize=word_tokenize,
                               init_token=SOS_TOKEN,
                               eos_token=EOS_TOKEN,
                               lower=True,
                               include_lengths=True,
                               batch_first=True)
        
        self.trg_field = Field(tokenize=word_tokenize,
                               init_token=SOS_TOKEN,
                               eos_token=EOS_TOKEN,
                               lower=True,
                               batch_first=True)

        fields_tuple = [(SRC_NAME, self.src_field), (TRG_NAME, self.trg_field)]

        # create toechtext datasets
        print('Create torchtext train...')
        self.train_data = self._make_torchtext_dataset(train, fields_tuple)
        print('Create torchtext validation...')
        self.valid_data = self._make_torchtext_dataset(val, fields_tuple)
        print('Create torchtext test...')
        self.test_data = self._make_torchtext_dataset(test, fields_tuple)

        # build vocabularies
        self.src_field.build_vocab(self.train_data, min_freq=1)
        self.trg_field.build_vocab(self.train_data, min_freq=1)
        print("i am field tuple",fields_tuple)

    def get_data(self):
        """Return train, validation and test data objects"""
        return self.train_data, self.valid_data, self.test_data

    def get_fields(self):
        """Return source and target field objects"""
        return self.src_field, self.trg_field

    def get_vocabs(self):
        """Return source and target vocabularies"""
        return self.src_field.vocab, self.trg_field.vocab

def prepare_data():
    
    train_m = train.copy()
    val_m = val.copy()
    test_m = test.copy()
    
    train_m = train_m.melt(id_vars=['id',"Question"],value_name="Answer")
    train_m = train_m[train_m['Answer'].astype(bool)].drop(['id','variable'],axis=1).values
    
    val_m = val_m.melt(id_vars=['id',"Question"],value_name="Answer")
    val_m = val_m[val_m['Answer'].astype(bool)].drop(['id','variable'],axis=1).values
    
    test_m = test_m.melt(id_vars=['id',"Question"],value_name="Answer")
    test_m = test_m[test_m['Answer'].astype(bool)].drop(['id','variable'],axis=1).values

    dataset = VerbalDataset(train_m,val_m,test_m)
    dataset.load_data_and_fields()
    src_vocab, trg_vocab = dataset.get_vocabs()
    train_data, valid_data, test_data = dataset.get_data()
    
    print('--------------------------------')
    print(f"After melting:")
    print(f"  Training data: {len(train_data.examples)}")
    print(f"  Evaluation data: {len(valid_data.examples)}")
    print(f"  Testing data: {len(test_data.examples)}")
    print('--------------------------------')
    print(f'Question example: {train_data.examples[2].src}\n')
    print(f'Answer example: {train_data.examples[2].trg}')
    print('--------------------------------')
    print(f"Unique tokens in questions vocabulary: {len(src_vocab)}")
    print(f"Unique tokens in all available answers vocabulary: {len(trg_vocab)}")
    print('--------------------------------')
    
    return dataset, src_vocab, trg_vocab, train_data, valid_data, test_data, train_m, val_m, test_m
