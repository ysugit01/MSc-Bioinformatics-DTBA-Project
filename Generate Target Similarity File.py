# Reference Code
# https://github.com/slavianap/Smith-Waterman-Algorithm/blob/master/Script.py

# Other reference codes:
# https://gist.github.com/radaniba/11019717
# https://github.com/Seb943/Smith_Waterman_Py/blob/master/Smith-Waterman.py

from enum import IntEnum
import pandas as pd
import numpy as np
import os


# Assigning the constants for the scores
class Score(IntEnum):
    MATCH = 2
    MISMATCH = -1
    GAP = -1


# Assigning the constant values for the traceback
class Trace(IntEnum):
    STOP = 0
    LEFT = 1
    UP = 2
    DIAGONAL = 3


def get_filenames(folder):
    path = folder
    dir_list = os.listdir(path)
    if 'Ignore' in dir_list:
        dir_list.remove('Ignore')
    return dir_list


# Reading the fasta file and keeping the formatted sequence's name and sequence
def fasta_reader(sequence_file):
    lines = open(sequence_file).readlines()
    sequence_name_row = lines[0][1:]
    sequence = ''
    for i in range(1, len(lines)):
        sequence = sequence + lines[i]
    return sequence_name_row.replace(" ", "").strip(), sequence.strip()


def read_fasta_files(folder_path, filenames):
    fasta_data = {}

    for filename in filenames:
        _, fasta = fasta_reader(folder_path + filename)
        key = filename[0:-6]
        fasta_data[key] = fasta

    return fasta_data


# Implementing the Smith Waterman local alignment
def smith_waterman(seq1, seq2):
    # Generating the empty matrices for storing scores and tracing
    row = len(seq1) + 1
    col = len(seq2) + 1
    matrix = np.zeros(shape=(row, col), dtype=np.int)
    tracing_matrix = np.zeros(shape=(row, col), dtype=np.int)

    # Initialising the variables to find the highest scoring cell
    max_score = -1
    max_index = (-1, -1)

    # Calculating the scores for all cells in the matrix
    for i in range(1, row):
        for j in range(1, col):
            # Calculating the diagonal score (match score)
            match_value = Score.MATCH if seq1[i - 1] == seq2[j - 1] else Score.MISMATCH
            diagonal_score = matrix[i - 1, j - 1] + match_value

            # Calculating the vertical gap score
            vertical_score = matrix[i - 1, j] + Score.GAP

            # Calculating the horizontal gap score
            horizontal_score = matrix[i, j - 1] + Score.GAP

            # Taking the highest score
            matrix[i, j] = max(0, diagonal_score, vertical_score, horizontal_score)

            # Tracking where the cell's value is coming from
            if matrix[i, j] == 0:
                tracing_matrix[i, j] = Trace.STOP

            elif matrix[i, j] == horizontal_score:
                tracing_matrix[i, j] = Trace.LEFT

            elif matrix[i, j] == vertical_score:
                tracing_matrix[i, j] = Trace.UP

            elif matrix[i, j] == diagonal_score:
                tracing_matrix[i, j] = Trace.DIAGONAL

                # Tracking the cell with the maximum score
            if matrix[i, j] >= max_score:
                max_index = (i, j)
                max_score = matrix[i, j]

    # Initialising the variables for tracing
    aligned_seq1 = ""
    aligned_seq2 = ""
    current_aligned_seq1 = ""
    current_aligned_seq2 = ""
    (max_i, max_j) = max_index

    # Tracing and computing the pathway with the local alignment
    while tracing_matrix[max_i, max_j] != Trace.STOP:
        if tracing_matrix[max_i, max_j] == Trace.DIAGONAL:
            current_aligned_seq1 = seq1[max_i - 1]
            current_aligned_seq2 = seq2[max_j - 1]
            max_i = max_i - 1
            max_j = max_j - 1

        elif tracing_matrix[max_i, max_j] == Trace.UP:
            current_aligned_seq1 = seq1[max_i - 1]
            current_aligned_seq2 = '-'
            max_i = max_i - 1

        elif tracing_matrix[max_i, max_j] == Trace.LEFT:
            current_aligned_seq1 = '-'
            current_aligned_seq2 = seq2[max_j - 1]
            max_j = max_j - 1

        aligned_seq1 = aligned_seq1 + current_aligned_seq1
        aligned_seq2 = aligned_seq2 + current_aligned_seq2

    # Reversing the order of the sequences
    aligned_seq1 = aligned_seq1[::-1]
    aligned_seq2 = aligned_seq2[::-1]

    return aligned_seq1, aligned_seq2, max_score


def pairwise_similarity(fasta_data):
    keys = list(fasta_data.keys())
    fasta_cnt = len(fasta_data)
    similarities = np.zeros((fasta_cnt, fasta_cnt))

    total = len(keys) * (len(keys) - 1) / 2 + len(keys)
    cnt = 0
    for i in range(len(keys)):
        for j in range(i + 1):
            _, _, max_score = smith_waterman(fasta_data[keys[i]], fasta_data[keys[j]])
            similarities[i, j] = max_score
            cnt += 1
            if(cnt%5 == 0):
                print("Progress = " + str(cnt) + ' of ' + str(total))

    for i in range(len(keys) - 1):
        similarities[i, :] = similarities[:, i]

    return similarities


if __name__ == "__main__":
    folder_path = 'data/GPCR/Fasta Files/'
    filenames = get_filenames(folder_path)
    fasta_data = read_fasta_files(folder_path, filenames)

    sim_targets = pd.DataFrame(data=pairwise_similarity(fasta_data),
                      index=fasta_data.keys(),
                      columns=fasta_data.keys())
    # print(sim_targets)
    # sim_targets.to_csv('data/GPCR/target_similarities.csv', sep=',')

    # https://academic.oup.com/bib/article/15/5/734/2422306?login=false
    sim_targets_norm = sim_targets.copy()
    for key1 in sim_targets.index:
        for key2 in sim_targets.index:
            sim_targets_norm[key1][key2] = sim_targets[key1][key2] / (sim_targets[key1][key1] ** 0.5 * sim_targets[key2][key2] ** 0.5)
    print(sim_targets_norm)
    sim_targets_norm.to_csv('data/GPCR/target_similarities_norm.csv', sep=',')



