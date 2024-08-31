from helpers.simple_dqn import SimpleDQN

#dqn = SimpleDQN('4x4')
#dqn.train(10000)
#dqn.run(10000)

dqn = SimpleDQN('8x8')
dqn.train(20000)
dqn.run(20000)