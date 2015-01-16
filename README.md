-------------------------------------------------------------------------------
Name:       A Hidden Markov Model
Purpose:    A class for the model, including parameters.
Author:     Haidong Zhang
Created:    28/08/2014
E-mail:     haidong_zhang13@163.com
-------------------------------------------------------------------------------
DataPreprocess.py (Firstly excute.)

Pre-process the data, generate two .txt file.
'artist.txt':
        artist_id: the artist id.
        num: the number of artists listened to.

'users_artists_timestamp.txt':
        user_id: the user id.
        artist_id: the artist id.
        timestamp: the time stamp.

------------------------------------------------------------------------------
HMMForCF
A Hidden Markov Model for Collaborative Filtering

-------------------------------------------------------------------------------
HMM.py

A class for modeling the hidden markov model.
HiddenStatesNum : The number of hidden states.
ObservationStatesNum: The number of observation states.
InitProbs: The probability of initial vector.
TransProbs: he probability of transition matrix.
Theta: The probability of emission matrix.
a, b: Represent the parameters of negative binomial distribution.

-------------------------------------------------------------------------------
HMM_ForwardBackward1.py

Forward-backward algorithm is used to train the model.

-------------------------------------------------------------------------------
CommonFunc.py

Some common used functions.

-------------------------------------------------------------------------------
UserModel.py

Model the users.

-------------------------------------------------------------------------------


.
