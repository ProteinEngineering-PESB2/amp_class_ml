import pandas as pd

class OneHotEncoder(object):

    def __init__(
            self, 
            dataset=None, 
            column_sequence=None,
            max_length=None) -> None:
        
        self.dataset = dataset
        self.column_sequence = column_sequence

        self.possible_residues = ['A', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'N', 'K', 'L', 'M', 'P', 'Q', 'R', 'S', 'T', 'V', 'W', 'Y']
        self.positions_residues = {'A' :0, 'C' : 1, 'D' : 2, 'E' : 3, 'F' : 4, 'G' : 5, 'H' : 6, 'I' : 7, 'N' : 8, 'K' : 9, 'L' : 10, 'M' : 11, 'P' : 12, 'Q' : 13, 'R' : 14, 'S' : 15, 'T' : 16, 'V' : 17, 'W' : 18, 'Y' : 19}

        self.max_length = max_length
    
    def __generate_vector_by_residue(self, residue):

        vector_coded = [0 for i in range(20)]
        vector_coded[self.positions_residues[residue]] = 1

        return vector_coded

    def __zero_padding(self, current_length):

        zero_padding_vector = [0 for i in range(current_length, self.max_length*20)]
        return zero_padding_vector

    def __coding_sequence(self, sequence):

        coded_vector = []

        for residue in sequence:
            coded_vector+=self.__generate_vector_by_residue(residue)

        if len(sequence) != self.max_length:
            coded_vector += self.__zero_padding(len(coded_vector))
        
        return coded_vector
    
    def run_process(self):

        matrix_coded = []

        for index in self.dataset.index:
            sequence = self.dataset[self.column_sequence][index]

            matrix_coded.append(
                self.__coding_sequence(sequence)
            )
        
        header = [f"p_{i}" for i in range(len(matrix_coded[0]))]
        df_coded = pd.DataFrame(data=matrix_coded, columns=header)

        return df_coded

