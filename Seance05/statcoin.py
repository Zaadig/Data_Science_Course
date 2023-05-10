import copy

from typing import *

from the_matrix import Coin,BiasedCoin

biased_coin = BiasedCoin()

def list_tosses(c:Coin,n:int)-> List[int]:
    return [c.toss() for _ in range(n)]


# From Seance 1,2,3

def estimate_bias(c:Coin,nb_tosses:int)->float:
    ''' estimate bias of coin c by the average of nb_tosses tosses.'''
    assert(nb_tosses > 0)
    return sum(c.toss() for _ in range(nb_tosses))/nb_tosses

def estimations_cfde(estimations:List[float])->Tuple[List[float],List[float]]:
    xs = copy.deepcopy(estimations)
    xs.sort()
    ys = [  (i+1)/len(estimations) for i in range(len(estimations))]
    return (xs,ys)

def average(floats:List[float])->float:
    assert(len(floats) > 0)
    return sum( f for f in floats)/len(floats)

def variance(floats:List[float])->float:
    assert(len(floats) > 1)
    mu = average(floats)
    return sum( (f-mu)**2 for f in floats[:-1])/len(floats[:-1])

def standard_deviation(floats:List[float])->float:
    assert(len(floats) > 1)
    return math.sqrt(variance(floats))
