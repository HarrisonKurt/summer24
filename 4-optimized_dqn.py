from helpers.optimized_dqn import OptimizedDQN

dqn = OptimizedDQN('8x8')
dqn.train(5000)
dqn.run(5000)