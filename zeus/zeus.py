# -*- coding: utf-8 -*-

import numpy as np
from song import Song

def main():
    """Main function of zeus."""

    # Read filenames of songs
    case1 = '../tests/case1.txt'
    song_list = import_playlist(case1)
    n = len(song_list)

    # Create objects
    for i in range(n):
        song_fname = song_list[i]
        song = Song(fname=song_fname)

        break


def import_playlist(path):
    """Imports the existing playlist."""

    song_list = np.genfromtxt(path, dtype='str')
    
    return song_list


if __name__ == "__main__":
    main()