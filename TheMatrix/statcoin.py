from typing import *

from the_matrix import Coin,BiasedCoin

biased_coin = BiasedCoin()

def list_tosses(c:Coin,n:int)-> List[int]:
    return [c.toss() for _ in range(n)]
