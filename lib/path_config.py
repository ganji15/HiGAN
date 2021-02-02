ImgHeight = 32

data_roots = {
    'iam_word': './data/'
}

data_paths = {
    'iam_word': {'trnval': 'trnvalset_words%d.hdf5'%ImgHeight,
                 'test': 'testset_words%d.hdf5'%ImgHeight},
    'iam_word_org': {'trnval': 'trnvalset_words%d_OrgSz.hdf5'%ImgHeight,
                     'test': 'testset_words%d_OrgSz.hdf5'%ImgHeight}
}
