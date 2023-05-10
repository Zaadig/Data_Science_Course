from typing import *
import random

class Coin:
    ''' Coin as a python object for statistical studies
    .toss() gives a random result 0 or 1.'''

    def __init__(self,bias=None,seed=None):
        ''' optional parameters:\
         - bias (float in [0,1]): probability of getting 1 in a toss,
         - seed (str or int): used to initialize pseudo-random number generation/'''
        if bias != None:
            assert(0 <= bias <= 1)
        self._bias = bias
        self._seed = seed
        if self._seed != None: # own random generator
            self.random_generator = random.Random(seed)
        else: # create random generator
            self.random_generator = random.Random()

    def toss_bias(self):
        ''' update bias to random.random() '''
        self._bias = random.random()

    def toss(self):
        assert( self._bias )
        return {False:0,True:1}[self.random_generator.random()<self._bias]

class PerfectCoin(Coin):

    def __init__(self,seed=None):
        Coin.__init__(self,bias=0.5,seed=seed)

    def toss(self):
        ''' return:
          0 with probability 0.5 
          1 with probability 0.5
          (bias = 0.5).'''
        return Coin.toss(self)

class BiasedCoin(Coin):
    
    def __init__(self,seed=None):
        bias = random.random()
        Coin.__init__(self,bias=bias,seed=seed)


    def set_bias(self,p):
        ''' set bias to given value '''
        self._bias = p


class MultiArmedBandit:
    ''' class for multi-armed bandits problem.
    .nb_arms:int the number of bandits
    ._arms:List[Coin] is a list of biased coins
    .toss(i:int):int return toss of the coin of index i in ._bandits
    .shuffle() suffles the list of coins ._bandits
    .toss_all_biases() redraw biases of all coins '''
    
    def __init__(self,nb_arms:int,seed=None):
        ''' initialisation just need the number of arms/coins.
        it is possible to add a seed for testing purpose.'''
        self.nb_arms = nb_arms
        if seed != None:
            bias_generator = random.Random(seed)
            biases = [ bias_generator.random() for _ in range(self.nb_arms)]
        else:
            biases = [ random.random() for _ in range(self.nb_arms)]
        self._arms = [ BiasedCoin(biases[i]) for i in range(self.nb_arms) ]

    def toss(self,i:int)->int:
        ''' tosses coin of index i, 0 <= i < .nb_bandits '''
        assert( 0 <= i < len(self._arms))
        return self._arms[i].toss()
         
    def shuffle(self):
        ''' shuffle all arms preserving their biases.
        XXX no seed version of shuffling '''
        random.shuffle(self._arms) 

    def toss_all_biases(self):
        ''' redraw the biases of all coins.
        XXX no seed version of toss of all biases '''
        for i in range(len(self._arms)):
            self._arms[i].toss_all_biases()

    def set_regular_but_shuffled_biases(self):
        ''' Biases are set to (9K+2i+1)/(20K)
        for i in range(k), for K = len(self._arms)
        then all bias are shuffled '''
        for i in range(len(self._arms)):
            K = len(self._arms)
            self._arms[i].set_bias((9*K+2*i+1)/(20*K))
        self.shuffle()


def time_length_flue_without_drug(): 
    ''' number of days of a flue without drug (or placebo) '''
    time = 1+int(3*random.random()) 
    while random.random() < 0.87:
        time += 1
    return time


def time_length_flue_with_drug(): 
    ''' number of days of a flue with drug '''
    time = 1+int(4*random.random())
    while random.random() < 0.82:
        time += 1
    return time


        
#### TESTS #############################
        
# perfect_coin = PerfectCoin(seed='a seed')
# print(f' perfect_coin_tosses {[perfect_coin.toss() for _ in range(10)]}')

# another_perfect_coin = PerfectCoin(seed='a seed')
# print(f' another_perfect_coin_tosses {[another_perfect_coin.toss() for _ in range(10)]}')

# biased_coin = BiasedCoin()
# print(f' biased_coin_tosses {[biased_coin.toss() for _ in range(10)]}')

    
