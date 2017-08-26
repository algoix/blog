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
    
    
    


