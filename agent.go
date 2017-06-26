package main

import (
	"github.com/NOX73/go-neural"
	"github.com/NOX73/go-neural/learn"
	"math/rand"
	"time"
)

type Agent struct {
	state_size     int
	action_size    int
	memory         []history
	gamma          float64
	epsilon        float64
	epsilon_min    float64
	epsilon_decay  float64
	learning_rate  float64
	model          neural.Network
	target_model   neural.Network
	HISTORY_LENGTH int
	DQN            bool
}

// init agent with default attributes
func (agent *Agent) init(state_size int, action_size int, dqn bool) Agent {
	agent.learning_rate = 0.95
	agent.action_size = action_size
	agent.state_size = state_size
	agent.epsilon = 1.0
	agent.epsilon_min = 0.01
	agent.epsilon_decay = 0.995
	agent.gamma = 0.95
	agent.HISTORY_LENGTH = 2000
	agent.memory = make([]history, agent.HISTORY_LENGTH)
	agent.model = *neural.NewNetwork(state_size, []int{state_size, state_size, action_size})
	agent.model.RandomizeSynapses()
	agent.target_model = *neural.NewNetwork(state_size, []int{state_size, state_size, action_size})
	agent.target_model.RandomizeSynapses()
	agent.DQN = dqn

	return *agent
}

// update target model after done game
func (agent *Agent) update_target_model() {
	agent.target_model = agent.model
}

// update model after get new reward value
func (agent *Agent) learn(input []float64, idealOut []float64) {
	model := &agent.model
	learn.Learn(model, input, idealOut, agent.learning_rate)
}

// predict action at state
func (agent *Agent) predict(input []float64) []float64 {
	return agent.model.Calculate(input)
}

// predict action at state plus 1
func (agent *Agent) predict_t_plus_1(input []float64) []float64 {
	return agent.target_model.Calculate(input)
}

// sample memory and update model
func (agent *Agent) learn_mini_batch(mini_batch_size int) {
	if len(agent.memory) >= agent.HISTORY_LENGTH {
		return
	}

	// sampling agent memory
	s := rand.NewSource(time.Now().Unix())
	r := rand.New(s) // initialize local pseudorandom generator
	mini_batch := make([]history, mini_batch_size)
	for i := 0; i < mini_batch_size; i++ {
		index := r.Intn(agent.HISTORY_LENGTH)
		mini_batch = append(mini_batch, agent.memory[index])
	}

	// update model for each history in mini batch
	for _, history := range mini_batch {
		state := history.state
		action := history.action
		reward := history.reward
		next_state := history.next_state
		done := history.done

		target := agent.predict(state)
		target_plus_1 := agent.predict_t_plus_1(next_state)
		max_q_t_plus_1 := -1.0
		for j := 0; j < len(target_plus_1); j++ {
			if max_q_t_plus_1 > target_plus_1[j] {
				max_q_t_plus_1 = target_plus_1[j]
			}
		}
		target[action] = (1.0-float64(done))*float64(agent.gamma)*max_q_t_plus_1 + float64(reward)
		agent.learn(state, target)
		//if agent.epsilon > agent.epsilon_min {
		//	agent.epsilon *= agent.epsilon_decay
		//}
		if !agent.DQN {
			agent.update_target_model()
		}
	}
}

func (agent *Agent) act(state []float64) int {
	action := 0

	return action
}

func (agent *Agent) remember(history history) {
	agent.memory = append(agent.memory, history)
}
