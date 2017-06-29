package main

import (
	"./nn"
	"fmt"
	tf "github.com/tensorflow/tensorflow/tensorflow/go"
	"github.com/tensorflow/tensorflow/tensorflow/go/op"
	"math/rand"
)

func main() {
	root := op.NewScope()
	net := nn.Init(root, 28, 28, 6)

	graph, err := root.Finalize()
	if err != nil {
		panic(err)
	}

	session, err := tf.NewSession(graph, &tf.SessionOptions{})
	if err != nil {
		panic(err)
	}

	array := make([]float32, 28*28)
	for i:=0; i<28*28; i++ {
		array[i] = rand.Float32()
	}
	input, err := tf.NewTensor(array)
	if err != nil {
		panic(nil)
	}
	//results, err := session.Run(map[tf.Output]*tf.Tensor{
	//	net.Input: input,
	//}, []tf.Output{net.Qout}, nil)
	//if err != nil {
	//	panic(err)
	//}
	//for _, result := range results {
	//	fmt.Println(result.Shape())
	//}

	target,err := tf.NewTensor(float32(1))
	if err != nil {
		panic(err)
	}
	act,err := tf.NewTensor(int32(2))
	if err != nil {
		panic(err)
	}
	results,err := session.Run(map[tf.Output]*tf.Tensor{
		net.Input: input,
		net.TargetQ: target,
		net.Actions: act,
	}, []tf.Output{net.Test}, []*tf.Operation{net.Var})
	if err != nil {
		panic(err)
	}
	for _, result := range results {
		fmt.Println(result.Value())
	}
}
