package main

import (
	"./nn"
	"fmt"
	tf "github.com/tensorflow/tensorflow/tensorflow/go"
	"github.com/tensorflow/tensorflow/tensorflow/go/op"
	"math/rand"
)

func main() {
	screen := int64(64)
	root := op.NewScope()
	net := nn.Init(root, screen, screen, 6, 1024)

	graph, err := root.Finalize()
	if err != nil {
		panic(err)
	}

	session, err := tf.NewSession(graph, &tf.SessionOptions{})
	if err != nil {
		panic(err)
	}

	array := make([]float32, screen*screen)
	for i:=0; i<int(screen*screen); i++ {
		array[i] = rand.Float32()
	}
	input, err := tf.NewTensor(array)
	if err != nil {
		panic(nil)
	}

	target,err := tf.NewTensor(float32(1))
	if err != nil {
		panic(err)
	}
	act,err := tf.NewTensor(int32(2))
	if err != nil {
		panic(err)
	}

	// run init tf.Variable operation
	results,err := session.Run(map[tf.Output]*tf.Tensor{
		net.Input: input,
		net.TargetQ: target,
		net.Actions: act,
	}, nil, []*tf.Operation{net.Init_var, net.Init_m, net.Init_v})
	if err != nil {
		panic(err)
	}

	// execute program
	results,err = session.Run(map[tf.Output]*tf.Tensor{
		net.Input: input,
		net.TargetQ: target,
		net.Actions: act,
	}, []tf.Output{net.Conv4}, nil)
	if err != nil {
		panic(err)
	}
	for _, result := range results {
		fmt.Println(result.Shape())
	}


}
