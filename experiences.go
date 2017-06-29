package main

import (
	"math/rand"
	"sort"
)

type Experiences struct {
	buffer      [][]float32
	buffer_size int
	i_counter   int
}

func (net *Experiences) Init(buffer_size int) {
	net.i_counter = 0
	net.buffer_size = buffer_size
	net.buffer = make([][]float32, 0)
}

func (exp *Experiences) Add(frame []float32) {
	if exp.i_counter < exp.buffer_size {
		exp.buffer = exp.buffer[1:]
	}
	exp.buffer = append(exp.buffer, frame)
	exp.i_counter += 1
}
func (exp *Experiences) Append(frames [][]float32) {
	if exp.i_counter+len(frames) < exp.buffer_size {
		drop := len(frames) - len(exp.buffer) - (exp.i_counter + 1)
		exp.buffer = exp.buffer[drop:]
	}
	exp.buffer = append(exp.buffer, frames...)
	exp.i_counter += len(frames)
}

func (exp *Experiences) Sample(size int) [][]float32 {
	buffer := make([][]float32, size)
	indices := make([]int, size)
	for i := 0; i < size; i++ {
		ind := rand.Int31n(int32(len(exp.buffer)))
		indices = append(indices, int(ind))
	}
	sort.Ints(indices)
	for i := 0; i < size; i++ {
		ind := indices[i]
		buffer = append(buffer, buffer[ind])
	}
	return buffer
}
