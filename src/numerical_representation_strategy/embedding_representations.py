from bio_embeddings.embed import (ProtTransBertBFDEmbedder, ProtTransAlbertBFDEmbedder,
                                 ProtTransT5BFDEmbedder, ProtTransT5UniRef50Embedder,
                                 ProtTransT5XLU50Embedder, ESM1bEmbedder, ProtTransXLNetUniRef100Embedder, BeplerEmbedder, CPCProtEmbedder,
                                  ESMEmbedder, ESM1vEmbedder, FastTextEmbedder,
                                  GloveEmbedder, OneHotEncodingEmbedder, PLUSRNNEmbedder, SeqVecEmbedder, Word2VecEmbedder)
import numpy as np
from tqdm import tqdm
import pandas as pd
from tqdm import TqdmWarning
import os

import warnings
warnings.filterwarnings("ignore", category=TqdmWarning)

class BioEmbeddings:
    """Apply Bio embeddings to a protein dataset"""
    def __init__(self, dataset, seq_column, is_reduced=True, device = None, path_export=None, column_response=None):
        self.dataset = dataset
        self.seq_column = seq_column
        self.column_response = column_response
        self.is_reduced=is_reduced
        self.device = device
        self.path_export = path_export

        self.embedder = None
        self.embeddings = None

    def __reducing(self):

        embedding = list(self.embeddings)
        matrix_embedding = [self.embedder.reduce_per_protein(e) for e in embedding]

        header = [f"p_{i}" for i in range(len(matrix_embedding[0]))]

        df_coded = pd.DataFrame(columns=header, data=matrix_embedding)
        return df_coded

    def __apply_model(self, model):

        if self.device is not None:
            self.embedder = model(device=self.device)
        else:
            self.embedder = model()
        
        self.embeddings = self.embedder.embed_many(self.dataset[self.seq_column].to_list())
        
        if self.is_reduced:
            return self.__reducing()
        else:
            return list(self.embeddings)

    def apply_esm1b(self, name_export):
        """Apply ESM1b embedder"""
        command = f"mkdir -p {self.path_export}esm1b"
        os.system(command)
        
        response = self.__apply_model(ESM1bEmbedder)
        response[self.column_response] = self.dataset[self.column_response]
        response.to_csv(f"{self.path_export}esm1b/{name_export}.csv", index=False)
    
    def apply_prottrans_bert(self, name_export):
        """Apply ProtTransBertBFD embedder"""

        command = f"mkdir -p {self.path_export}protrans_bert"
        os.system(command)

        response =  self.__apply_model(ProtTransBertBFDEmbedder)

        response[self.column_response] = self.dataset[self.column_response]
        response.to_csv(f"{self.path_export}protrans_bert/{name_export}.csv", index=False)

    def apply_prottrans_albert(self, name_export):
        """Apply ProtTransBertBFD embedder"""

        command = f"mkdir -p {self.path_export}protrans_albert"
        os.system(command)

        response= self.__apply_model(ProtTransAlbertBFDEmbedder)
        
        response[self.column_response] = self.dataset[self.column_response]
        response.to_csv(f"{self.path_export}protrans_albert/{name_export}.csv", index=False)

    def apply_prottrans_t5_uniref(self, name_export):
        """Apply ProtTransT5UniRef50 embedder"""

        command = f"mkdir -p {self.path_export}protrans_uniref"
        os.system(command)

        response = self.__apply_model(ProtTransT5UniRef50Embedder)

        response[self.column_response] = self.dataset[self.column_response]
        response.to_csv(f"{self.path_export}protrans_uniref/{name_export}.csv", index=False)

    def apply_prottrans_t5_xlu50(self, name_export):
        """Apply ProtTransT5XLU50 embedder"""

        command = f"mkdir -p {self.path_export}protrans_xlu50"
        os.system(command)

        response =  self.__apply_model(ProtTransT5XLU50Embedder)

        response[self.column_response] = self.dataset[self.column_response]
        response.to_csv(f"{self.path_export}protrans_xlu50/{name_export}.csv", index=False)

    def apply_prottrans_t5bdf(self, name_export):
        """Apply ProtTransT5BFD embedder"""

        command = f"mkdir -p {self.path_export}protrans_bdf"
        os.system(command)

        response =  self.__apply_model(ProtTransT5BFDEmbedder)

        response[self.column_response] = self.dataset[self.column_response]
        response.to_csv(f"{self.path_export}protrans_bdf/{name_export}.csv", index=False)

    def apply_prottrans_xlnet(self, name_export):
        """Apply ProtTransXLNET embedder"""

        command = f"mkdir -p {self.path_export}protrans_xlnet"
        os.system(command)

        response =  self.__apply_model(ProtTransXLNetUniRef100Embedder)

        response[self.column_response] = self.dataset[self.column_response]
        response.to_csv(f"{self.path_export}protrans_xlnet/{name_export}.csv", index=False)

    def apply_bepler(self, name_export):
        """Apply BeplerEmbedder embedder"""

        command = f"mkdir -p {self.path_export}bepler"
        os.system(command)

        response =  self.__apply_model(BeplerEmbedder)

        response[self.column_response] = self.dataset[self.column_response]
        response.to_csv(f"{self.path_export}bepler/{name_export}.csv", index=False)

    def apply_cpc_prot(self, name_export):
        """Apply CPCProtEmbedder embedder"""

        command = f"mkdir -p {self.path_export}cpcprot"
        os.system(command)

        response =  self.__apply_model(CPCProtEmbedder)

        response[self.column_response] = self.dataset[self.column_response]
        response.to_csv(f"{self.path_export}cpcprot/{name_export}.csv", index=False)

    def apply_esme(self, name_export):
        """Apply ESMEmbedder embedder"""

        command = f"mkdir -p {self.path_export}esme"
        os.system(command)

        response =  self.__apply_model(ESMEmbedder)

        response[self.column_response] = self.dataset[self.column_response]
        response.to_csv(f"{self.path_export}esme/{name_export}.csv", index=False)

    def apply_esme1v(self, name_export):
        """Apply ESM1vEmbedder embedder"""

        command = f"mkdir -p {self.path_export}esme1v"
        os.system(command)

        response =  self.__apply_model(ESM1vEmbedder)

        response[self.column_response] = self.dataset[self.column_response]
        response.to_csv(f"{self.path_export}esme1v/{name_export}.csv", index=False)

    def apply_fasttextv(self, name_export):
        """Apply FastTextEmbedder embedder"""

        command = f"mkdir -p {self.path_export}fasttext"
        os.system(command)

        response =  self.__apply_model(FastTextEmbedder)

        response[self.column_response] = self.dataset[self.column_response]
        response.to_csv(f"{self.path_export}fasttext/{name_export}.csv", index=False)

    def apply_glove(self, name_export):
        """Apply GloveEmbedder embedder"""

        command = f"mkdir -p {self.path_export}glove"
        os.system(command)

        response =  self.__apply_model(GloveEmbedder)

        response[self.column_response] = self.dataset[self.column_response]
        response.to_csv(f"{self.path_export}glove/{name_export}.csv", index=False)

    def apply_onehot(self, name_export):
        """Apply OneHotEncodingEmbedder embedder"""

        command = f"mkdir -p {self.path_export}onehot"
        os.system(command)

        response =  self.__apply_model(OneHotEncodingEmbedder)

        response[self.column_response] = self.dataset[self.column_response]
        response.to_csv(f"{self.path_export}onehot/{name_export}.csv", index=False)

    def apply_plusrnn(self, name_export):
        """Apply PLUSRNNEmbedder embedder"""

        command = f"mkdir -p {self.path_export}plusrnn"
        os.system(command)

        response =  self.__apply_model(PLUSRNNEmbedder)

        response[self.column_response] = self.dataset[self.column_response]
        response.to_csv(f"{self.path_export}plusrnn/{name_export}.csv", index=False)

    def apply_seqvec(self, name_export):
        """Apply SeqVecEmbedder embedder"""

        command = f"mkdir -p {self.path_export}seqvec"
        os.system(command)

        response =  self.__apply_model(SeqVecEmbedder)

        response[self.column_response] = self.dataset[self.column_response]
        response.to_csv(f"{self.path_export}seqvec/{name_export}.csv", index=False)

    def apply_word2vec(self, name_export):
        """Apply Word2VecEmbedder embedder"""

        command = f"mkdir -p {self.path_export}word2vec"
        os.system(command)

        response =  self.__apply_model(Word2VecEmbedder)

        response[self.column_response] = self.dataset[self.column_response]
        response.to_csv(f"{self.path_export}word2vec/{name_export}.csv", index=False)