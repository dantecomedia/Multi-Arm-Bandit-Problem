import matplotlib.pyplot as plt 
import numpy as np 

NUM_TRAILS=10000
EPS=0.1
BANDIT_PROBABILITIES=[0.2,0.5,0.75]


class Bandit:
	def __init__(self,p):
		self.p=p
		self.p_estimate=0.
		self.N=0.


	def pull(self):
		return np.random.random() <self.p



	def update(self,x):

		self.N+=1.
		self.p_estimate=((self.N-1)*self.p_estimate+x) /self.N




def experiment():

	bandits=[Bandit(p) for p in BANDIT_PROBABILITIES]

	rewards =np.zeros(NUM_TRAILS)

	num_times_explored=0
	num_times_exploited=0
	num_optimal=0

	optimal_j=np.argmax([b.p for b in bandits])

	print("optimal j:", optimal_j)

	for i in range(NUM_TRAILS):
		if np.random.random()<EPS:
			num_times_explored+=1
			j=np.random.randint(len(bandits))

		else:
			num_times_exploited+=1
			j=np.argmax([b.p_estimate for b in bandits])

		if j == optimal_j:
			num_optimal+=1

		x=bandits[j].pull()

		rewards[i]=x

		bandits[j].update(x)




	for b in bandits:
		print("mean estimate:", b.p_estimate)


	print("total reward earned :", rewards.sum())
	print("overall win rate :", rewards.sum()/NUM_TRAILS)
	print("number of time explored :", num_times_explored)
	print("number of time exploited:", num_times_exploited)
	print("number of time selected optimal bandit: ", num_optimal)



	cumulative_rewards=np.cumsum(rewards)
	win_rates=cumulative_rewards/(np.arange(NUM_TRAILS)+1)

	plt.plot(win_rates)
	plt.plot(np.ones(NUM_TRAILS)*np.max(BANDIT_PROBABILITIES))
	plt.show()



if __name__=="__main__":
	experiment()
