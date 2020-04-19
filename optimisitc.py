from __future__ import print_function , division 
from builtins import range 



import matplotlib.pyplot as plt 
import numpy as np 



NUM_TRAILS = 10000
EPS=0.1
BANDIT_PROBABILITIES =[0.2,0.5,0.75]


class Bandit:
	def __init__(self, p):
		self.p=p
		self.p_estimate=5.
		self.N=1.

	def pull(self):
		return np.random.random() <self.p 


	def update(self,x):
		self.N+=1
		self.p_estimate=((self.N-1)*self.p_estimate+x) /self.N 


def experiment():

	bandits=[Bandit(p) for p in BANDIT_PROBABILITIES]
	rewards = np.zeros(NUM_TRAILS)

	for i in range(NUM_TRAILS):

		j=np.argmax([b.p_estimate for b in bandits])
		x=bandits[j].pull()

		rewards[i]=x

		bandits[j].update(x)


	for b in bandits:
		print("mean estimate:", b.p_estimate)

		print("total reward earned",rewards.sum())
		print("overal win rate:", rewards.sum()/NUM_TRAILS)
		print("num times selected each bandit",[b.N for b in bandits])

		cumulative_rewards = np.cumsum(rewards)
		win_rates=cumulative_rewards/(np.arange(NUM_TRAILS)+1)

		plt.ylim([0,1])
		plt.plot(win_rates)
		plt.plot(np.ones(NUM_TRAILS)*np.max(BANDIT_PROBABILITIES))
		plt.show()



if __name__=="__main__":
	experiment()








