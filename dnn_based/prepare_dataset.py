import musdb
import numpy as np
import h5py
import librosa as lsa

def prepareDataset(tcontext, fft_size, musdb18_path):
    mus = musdb.DB(root_dir=musdb18_path)
    # load the training tracks pointers
    train_tracks = mus.load_mus_tracks(subsets=['train'])

    # load the test tracks pointers
    test_tracks = mus.load_mus_tracks(subsets=['test'])

    # open file
    with h5py.File(musdb18_path + 'musdb_classify.hdf5', 'w') as hdf:
        # with h5py.File('/media/archive/' + 'musdb_classify_tcontext_' + str(tcontext) + '.hdf5', 'w') as hdf:
        # store hyperparameters
        hdf.attrs.create('tcontext', tcontext)
        # initialize frequency index variable
        f_starts = np.array([], dtype='intp')
        f_starts = np.hstack((f_starts, (0)))
        # We start two counters: for each file and for each STFT frame
        chunk_ind = 0
        n = 0
        # initialize variables for stats
        mean = np.zeros(int(fft_size / 2 + 1), dtype=np.float32)
        M2 = np.zeros(int(fft_size / 2 + 1), dtype=np.float32)

        # initialize inputs and labels datasets. Uses chunking based on tcontext
        hdf.create_dataset('track_mag', (1, 1), maxshape=(None, int(fft_size / 2 + 1)),
                           chunks=(tcontext, int(fft_size / 2 + 1)), dtype='f')
        hdf.create_dataset('label', (1, 1), maxshape=(None, None), chunks=True, dtype=int)
        # Computed with the online welford algorithm
        hdf.create_dataset('mean', (1, int(fft_size / 2 + 1)), 'f')  # binwise statics
        hdf.create_dataset('std', (1, int(fft_size / 2 + 1)), 'f')


        # Define insert function for each track
        def insert_track(track, tcontext, n, mean, M2, f_starts, label, chunk_ind):
            # Cast to double precision
            track = np.float32(track)
            # Find out how much padding is needed so STFS fit tcontext hdf5-chunking
            padding = len(track)
            while ((padding / fft_size) * int(fft_size / int(fft_size / 4)) + 1) % tcontext != 0:
                padding += 1
            track = np.hstack((track, np.zeros((padding - len(track)), dtype=np.float32)))

            # compute STFT using librosa
            track = np.abs(
                lsa.core.stft(track, n_fft=fft_size, hop_length=int(fft_size / 4), win_length=None, window='hann',
                              center=True, dtype='complex64', pad_mode='reflect'))
            track = np.swapaxes(track, 0, 1)
            # Apply logarithmic compression
            track = np.log10(1+track)
            # Store input and label in HDF5
            f_starts = np.hstack((f_starts, (len(track) + f_starts[-1])))  # we store frequency index
            hdf['track_mag'].resize((f_starts[-1], int(fft_size / 2 + 1)))
            hdf['track_mag'][f_starts[-2]:f_starts[-1], :] = track

            chunk_ind += int(len(track) / tcontext)

            hdf['label'].resize((chunk_ind, 1))
            hdf['label'][chunk_ind - int(len(track) / tcontext):chunk_ind] = np.expand_dims(
                np.tile(label, int(len(track) / tcontext)), 1)

            # Then do online STD using welford's algorithm
            for timeframe in range(len(track)):
                # Iterate over time STFT
                n += 1
                delta = track[timeframe, :] - mean
                mean = mean + delta / n
                M2 = M2 + delta * (track[timeframe, :] - mean)

            return n, mean, M2, f_starts, chunk_ind


        # Run for each track in the dataset: get audio, downmix to mono, insert to dataset
        for track in train_tracks:
            aux = track.targets['vocals'].audio
            vocals = 0.5 * aux[:, 0] + 0.5 * aux[:, 1]
            n, mean, M2, f_starts, chunk_ind = insert_track(vocals, tcontext, n, mean, M2, f_starts, 0, chunk_ind)
            aux = track.targets['drums'].audio
            drums = 0.5 * aux[:, 0] + 0.5 * aux[:, 1]
            n, mean, M2, f_starts, chunk_ind = insert_track(drums, tcontext, n, mean, M2, f_starts, 1, chunk_ind)
            aux = track.targets['bass'].audio
            bass = 0.5 * aux[:, 0] + 0.5 * aux[:, 1]
            n, mean, M2, f_starts, chunk_ind = insert_track(bass, tcontext, n, mean, M2, f_starts, 2, chunk_ind)
            aux = track.targets['other'].audio
            other = 0.5 * aux[:, 0] + 0.5 * aux[:, 1]
            n, mean, M2, f_starts, chunk_ind = insert_track(other, tcontext, n, mean, M2, f_starts, 3, chunk_ind)

            if chunk_ind % 5 == 0:
                print(str((len(test_tracks) + len(train_tracks)) * 4 - (chunk_ind)) + ' files remaining...')

        for track in test_tracks:
            aux = track.targets['vocals'].audio
            vocals = 0.5 * aux[:, 0] + 0.5 * aux[:, 1]
            n, mean, M2, f_starts, chunk_ind = insert_track(vocals, tcontext, n, mean, M2, f_starts, 0, chunk_ind)
            aux = track.targets['drums'].audio
            drums = 0.5 * aux[:, 0] + 0.5 * aux[:, 1]
            n, mean, M2, f_starts, chunk_ind = insert_track(drums, tcontext, n, mean, M2, f_starts, 1, chunk_ind)
            aux = track.targets['bass'].audio
            bass = 0.5 * aux[:, 0] + 0.5 * aux[:, 1]
            n, mean, M2, f_starts, chunk_ind = insert_track(bass, tcontext, n, mean, M2, f_starts, 2, chunk_ind)
            aux = track.targets['other'].audio
            other = 0.5 * aux[:, 0] + 0.5 * aux[:, 1]
            n, mean, M2, f_starts, chunk_ind = insert_track(other, tcontext, n, mean, M2, f_starts, 3, chunk_ind)

            if chunk_ind % 5 == 0:
                print(str((len(test_tracks) + len(train_tracks)) * 4 - (chunk_ind)) + ' files remaining...')

        # Finally store statics in HDF5
        hdf['mean'][0, ...] = mean
        hdf['std'][0, ...] = np.sqrt(M2 / (n - 1))

        # Also store indexes of frequency data streams: index corresponds to utterance starting point
        hdf.create_dataset('f_indexes', (len(f_starts),), dtype='intp')
        hdf['f_indexes'][...] = f_starts
        # Close the HDF5 file
        hdf.close()
        print('Done')