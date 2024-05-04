from bio_embeddings.embed import (ProtTransBertBFDEmbedder, ProtTransAlbertBFDEmbedder,
                                 ProtTransT5BFDEmbedder, ProtTransT5UniRef50Embedder,
                                 ProtTransT5XLU50Embedder, ESM1bEmbedder, ProtTransXLNetUniRef100Embedder)
import numpy as np
from tqdm import tqdm
import pandas as pd
from tqdm import TqdmWarning
import warnings
warnings.filterwarnings("ignore", category=TqdmWarning)

class BioEmbeddings:
    """Apply Bio embeddings to a protein dataset"""
    def __init__(self, dataset, seq_column, is_reduced=True, device = None):
        self.dataset = dataset
        self.seq_column = seq_column
        self.is_reduced=is_reduced
        self.device = device
        self.embedder = None
        self.embeddings = None

        self.data_coded = None

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

    def apply_esm1b(self):
        """Apply ESM1b embedder"""
        return self.__apply_model(ESM1bEmbedder)
    
    def apply_prottrans_bert(self):
        """Apply ProtTransBertBFD embedder"""
        return self.__apply_model(ProtTransBertBFDEmbedder)
    
    def apply_prottrans_albert(self):
        """Apply ProtTransBertBFD embedder"""
        return self.__apply_model(ProtTransAlbertBFDEmbedder)
        
    def apply_prottrans_t5_uniref(self):
        """Apply ProtTransT5UniRef50 embedder"""
        return self.__apply_model(ProtTransT5UniRef50Embedder)
    
    def apply_prottrans_t5_xlu50(self):
        """Apply ProtTransT5XLU50 embedder"""
        return self.__apply_model(ProtTransT5XLU50Embedder)
    
    def apply_prottrans_t5bdf(self):
        """Apply ProtTransT5BFD embedder"""
        return self.__apply_model(ProtTransT5BFDEmbedder)
    
    def apply_prottrans_xlnet(self):
        """Apply ProtTransXLNET embedder"""
        return self.__apply_model(ProtTransXLNetUniRef100Embedder)