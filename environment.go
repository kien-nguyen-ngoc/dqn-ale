package main

type environment struct {
}

func (e *environment) init(name string) {

}

func (e *environment) get_action_space() int {
	return 0
}

func (e *environment) get_observation_space() int {
	return 0
}

func (e *environment) step(action int) history {
	h := new(history)

	return *h
}

func (e *environment) reset() []float64 {
	return make([]float64, 10)
}
