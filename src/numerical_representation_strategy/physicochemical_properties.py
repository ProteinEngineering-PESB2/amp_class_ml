import pandas as pd

class PhysicochemicalEncoder:

    def __init__(self,
                 dataset=None,
                 sep_dataset=",",
                 property_encoder="Group_0",
                 dataset_encoder=None,
                 name_column_seq="sequence",
                 columns_to_ignore=[]):

        self.dataset = dataset
        self.sep_dataset = sep_dataset

        self.property_encoder = property_encoder
        self.dataset_encoder = dataset_encoder
        self.name_column_seq = name_column_seq
        self.columns_to_ignore = columns_to_ignore

        self.possible_residues = [
            'A',
            'C',
            'D',
            'E',
            'F',
            'G',
            'H',
            'I',
            'N',
            'K',
            'L',
            'M',
            'P',
            'Q',
            'R',
            'S',
            'T',
            'V',
            'W',
            'Y'
        ]

        self.df_data_encoded = None

        self.status = False
        self.message= ""

    def run_process(self):
        self.__make_validations()

        if self.status == True:
            self.zero_padding = self.__check_max_size()
            self.__encoding_dataset()
            self.message = "ENCODING OK"
        
    def __check_columns_in_df(
            self,
            check_columns=None,
            columns_in_df=None):

        response_check = True

        for colum in check_columns:
            if colum not in columns_in_df:
                response_check=False
                break
        
        return response_check
    
    def __make_validations(self):

        # read the dataset with encoders
        self.dataset_encoder.index = self.dataset_encoder['residue']
        
        # check input dataset
        if self.name_column_seq in self.dataset.columns:
            
            if isinstance(self.columns_to_ignore, list):

                if len(self.columns_to_ignore)>0:
                    
                    response_check = self.__check_columns_in_df(
                        columns_in_df=self.dataset.columns.values,
                        check_columns=self.columns_to_ignore
                    )
                    if response_check == True:
                        self.status=True
                    else:
                        self.message = "ERROR: IGNORE COLUMNS NOT IN DATASET COLUMNS"   
                else:
                    pass
            else:
                self.message = "ERROR: THE ATTRIBUTE columns_to_ignore MUST BE A LIST"
        else:
            self.message = "ERROR: COLUMN TO USE AS SEQUENCE IS NOT IN DATASET COLUMNS"    

    def __check_residues(self, residue):
        if residue in self.possible_residues:
            return True
        else:
            return False

    def __encoding_residue(self, residue):

        if self.__check_residues(residue):
            return self.dataset_encoder[self.property_encoder][residue]
        else:
            return False

    def __check_max_size(self):
        size_list = [len(seq) for seq in self.dataset[self.name_column_seq]]
        return max(size_list)

    def __encoding_sequence(self, sequence):

        sequence = sequence.upper()
        sequence_encoding = []

        for i in range(len(sequence)):
            residue = sequence[i]
            response_encoding = self.__encoding_residue(residue)
            if response_encoding != False:
                sequence_encoding.append(response_encoding)

        # complete zero padding
        for k in range(len(sequence_encoding), self.zero_padding):
            sequence_encoding.append(0)

        return sequence_encoding

    def __encoding_dataset(self):

        #print("Start encoding process")
        if len(self.columns_to_ignore)>0:
            df_columns_ignore = self.dataset[self.columns_to_ignore]
            dataset_to_encode = self.dataset.drop(columns=self.columns_to_ignore)
        else:
            df_columns_ignore=None
            dataset_to_encode = self.dataset

        print("Encoding and Processing results")

        matrix_data = []
        for index in dataset_to_encode.index:
            sequence_encoder = self.__encoding_sequence(sequence=dataset_to_encode[self.name_column_seq][index])
            matrix_data.append(sequence_encoder)

        print("Creating dataset")
        header = ['p_{}'.format(i) for i in range(len(matrix_data[0]))]
        print("Export dataset")

        self.df_data_encoded = pd.DataFrame(matrix_data, columns=header)

        if len(self.columns_to_ignore)>0:
            self.df_data_encoded = pd.concat([self.df_data_encoded, df_columns_ignore], axis=1) 