# RL
## Intro
### S,A,R
#### States: Holding long, holding short, technical value (RSI, BB), return per minute
#### Actions: Buy, Sell, Cover, Short, do nothing
#### Rewards: Return from trade, return per 5 minutes

1.  Markov Decision Problems: states (s), actions (a), transition function T[s,a,s], Reward function R[s,a]. Reward-> $
2.  Find: optimal policy that will maximize reward.
3.  Algorithm-> Policy iteration and value iteration.
4.  most of the time T and R are unknown.
### Model based RL: Build a model T[s,a,s'] over time  and R[s,a]. We can apply value/policy iteration 
### Model Free or Q Learning: sum of gamma discounted rewards 
## Q Learning

    Q[s,a]=immediate learning+discounted reward
    Pi[s]=argmax<sub>a</sub>(Q[s,a]) -> converging tp -> Pi* and Q*[s,a]
### Building Q table
#### Q Learning procedure
    1. DataSet: train, test
    2. s,pi(S)->a||->s':r||<s,a,s':r>
    3. iterate over time <s,a,s':r>
    4. test policy
    5. repeat until converge
    Details:
    1. Set start time,init Q[]
    2. compute s
    3. select a
    4. observe r,s'
    5. update Q
    note: 1 to 4 is <s,a,s':r>
#### update rule. 
    Q'[s,a]=(1-alpha)Q[s,a]+alpha*improved estimate
           =(1-alpha)Q[s,a]+alpha*(r+gamma*later rewards)
           =(1-alpha)Q[s,a]+alpha*(r+gamma*Q[s',argmax<sub>a'</sub>(Q[s',a'])])
    note: alpha=learning rate and gamma=discounting rate
    
    The formula for computing Q for any state-action pair <s, a>, given an experience tuple <s, a, s', r>, 
    is:Q'[s, a] = (1 - α) · Q[s, a] + α · (r + γ · Q[s', argmaxa'(Q[s', a'])])
    Here:
	•	r = R[s, a] is the immediate reward for taking action a in state s,
	•	γ &in; [0, 1] (gamma) is the discount factor used to progressively reduce the value of future rewards,
	•	s' is the resulting next state,
	•	argmaxa'(Q[s', a']) is the action that maximizes the Q-value among all possible actions a' from s', and,
	•	α &in; [0, 1] (alpha) is the learning rate used to vary the weight given to new experiences compared with past Q-values.
    
#### Creating the state
    State is an integer
    Discretize each factor
    Combine
#### Discretize
    stepsize=size(data)/steps
    data.sort()
    for i in range (0,steps):
        threshold[i]=data[(i+1)*stepsize]# grouping at every 'stepsize' chunk into an integer 
---------------------------------------- 
 
- suppose you want to discretize inputs into a range [0,N] then you have N steps 
- grouping at every 'stepsize' chunk into an integer 

---------------------------------------- 
 
- suppose you want to discretize inputs into a range [0,N] then you have N steps 	
#### Dyna: a special Q-learner 
 
a problem with Q-learners is it takes too many experience tupuls to converge. 
i.e. too many real world interactions, but we don't want to let the learner have too much real trading just for practice. 
 
Dyna builds models of transition T, and a reward matrix R, so it hallucinates the learner to have many (a few hundred) interactions (after a afew real interactions), to update Q table. 
 
## 
## a normal Q-learner  (recap) 
## 
1. init Q[] 
2. observe s 
3. execute a, observe s' & r 
4. update Q with <s,a,s',r> 
5. repeat 2,3,4 until converge (expensive operation) 
 
## 
## Dyna-Q 
## 
1. init Q[] 
2. observe s 
3. execute a, observe s' & r 
4. update Q with <s,a,s',r> # not significant 
5. repeat 2,3,4 a few times, to learn models of T & R 
6. hallucinate 2,3 with T & R 
7. update Q with <s,a,s',r> 
8. repeat 6,7 until converge (cheaper operation !) 
 
T[s,a,s'] = probability that taking a in s ends up in s' 
R[s,a] = expected reward you get for taking a in s 
 
 
# 
# how to hallucinate  (step 6) 
# 
-- keep track of encountered state+action pairs) e.g. [(s0,a0),(s1,a1),,,,(sN,aN)] 
--- it's possible to get repeated state+action pairs. 
--- yes multiple actions may be associated with a state, because you can come to the same state as before and take a diff action. 
 
 s = randomly chosen from known(previously visited) states 
 a = randomly chosen from known(previously taken) actions for that random s you just chose 
 s'= infer from T[] 
 r = R[s,a] 
 
how do i infer s' from T[] ?  (assume you have the above described T[]) 
(1)  s'= np.argmax(T[s,a,:]) 
-- not necessarily good to blindly take the highest probability s_prime because if it's not the effective state to be in, it takes long to converge. 
(2) s' = np.random.choice(range(self.num_states),p=self.T[random_s,random_a,:]) 
-- this is better. 
 
# 
# how to update T & R   # (step 5) 
# 
- learning a model of T 
-- just count real world examples, and build a table Tc 
 init Tc[all] = 0.00001 
 while executing, observe s,a,s' 
 increment Tc[s,a,s'] 
 
 T[s,a,s'] = Tc[s,a,s'] / Sum(Tc[s,a,i]) 
                           i 
 
- learning a model of R 
 R[s,a] = expected reward for s,a 
 r = immediate reward 
 R'[s,a] = (1-A)*R[s,a] + A*r      # Alpha usually is 0.2 (this alpha can be diff from the alpha for Q table) 
 
 
NOTE: there are a few variants to dyna-Q implementation. the point is to facilitate the hallucinated learning. 
(ref) https://github.com/paulorauber/rl/blob/master/learning/model_building.py   # sample code 
(ref) https://webdocs.cs.ualberta.ca/~sutton/book/ebook/node96.html     # theory 
(ref) http://www-anw.cs.umass.edu/~barto/courses/cs687/Chapter%209.pdf  # theory 
 
 create a Tc table ("T count") that counts the number of times each subsequent state is reached after taking an action in a given state. Initialize all values to 0.00001 to avoid division by 0 errors. While executing QLearning, observe s,a,s′ and increment Tc[s,a,s′]

.

Formula for calculating the probability of each s' from [s,a]

Tc[s,a,s′]∑iTc[s,a,i]

    The numerator is the count of times s,a lead to each s'.
    The denominator is the sum of Tc[s,a,:]

    (normalizes each s' to its probability)

Dyna-Q then updates a state's expected reward as the weighted sum of the reward in each s' weighted by their relative probability.
 
################# 
##  appendix   ## 
################# 
 
work station: buffet03.cc.gatech.edu 
(ref) http://quantsoftware.gatech.edu/ML4T_Software_Setup 
 
you basically copy git repo locally then do your coding, then test, then maybe WinSCP and submit your python code via t-square 
 
scp -r <username>@buffetbuffet03.cc.gatech.edu:/path/to/directory /path/to/destination 
(or use WinSCP) 
 
for IDE, anaconda/spyder will be good. 
 
# 
# git memo 
# 
git clone https://github.gatech.edu/tb34/ML4T_2016Fall.git 
git pull   # to sync with the latest 
 
## 
## spyder 
## 
 
http://stackoverflow.com/questions/26679272/not-sure-how-to-use-argv-with-spyder 
==> alternatively you can do the below on the iphython console 
runfile('C:/Users/mel/Desktop/gatech/ml4t/ML4T_2016Fall/mc3_p1/testlearner.py', args='/Users/mel/Desktop/gatech/ml4t/ML4T_2016Fall/mc3_p1/Data/simple.csv', wdir='C:/Users/mel/Desktop/gatech/ml4t/ML4T_2016Fall/mc3_p1') 
 
 
## 
##  wall street lingo 
## 
- cyclical: means a stock, a company's business performance generally depends on the economy, like US steel. 
- secular: regression-resistant stock, like food and medicine, tooth brush company whose performance is consistent regardless of overall economy health 
-- know a stock is either a cyclical or secular growth stock. when the economy is down buy secular, when up buy cyclical. 
- rotation: a term often used when money flows between cyclical and secular stocks. 
 
 
### 
### what hedge funds really do 
### 
 
headge funds VS mutual funds 
H: less regulated. 
M: more regulated. has to declare strategy in prospectus. can advertise and access small investors. 

## References
https://github.com/paulorauber/rl/tree/master/learning
