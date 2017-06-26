package main

import (
	"flag"
	"fmt"
	"os"
)

func main() {



	return

	var name string
	flag.StringVar(&name, "name", "", "game name")
	flag.Parse()
	// init environment
	env := new(environment)
	env.init(name)

	// init agent
	agent := new(Agent)
	agent.init(env.get_observation_space(), env.get_action_space(), true)

	// train
	EPISODES := 5000
	batch_size := 64

	for e := 0; e < EPISODES; e++ {
		state := env.reset()
		//state = env.reshape(state, agent.state_size)
		for time := 0; time < 500; time++ {

			//env.render()
			action := agent.act(state)
			history := env.step(action)
			agent.remember(history)
			if history.done != 0 {
				agent.update_target_model()
				fmt.Fprintf(os.Stdout, "episode: %d/%d, score: %d, e: %.2f", e, EPISODES, time, agent.epsilon)
				break
			}
		}
		if len(agent.memory) > batch_size {
			agent.learn_mini_batch(batch_size)
		}
	}
}
