package main

import (
	"./nn"
	"fmt"
	tf "github.com/tensorflow/tensorflow/tensorflow/go"
	"github.com/tensorflow/tensorflow/tensorflow/go/op"
	"math"
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

	// prepare input
	//input, err := tf.NewTensor([1][5][5][1]float32{{
	//	{
	//		{1}, {3}, {4}, {6}, {7},
	//	},
	//	{
	//		{7}, {0}, {2}, {8}, {9},
	//	},
	//	{
	//		{1}, {3}, {7}, {5}, {1},
	//	},
	//	{
	//		{5}, {2}, {6}, {9}, {3},
	//	},
	//	{
	//		{8}, {6}, {0}, {7}, {9},
	//	},
	//}})
	array := make([]float32, 28*28)
	for i:=0; i<28*28; i++ {
		array[i] = float32(math.Sqrt(float64(i)))
	}
	input, err := tf.NewTensor(array)
	if err != nil {
		panic(nil)
	}

	results, err := session.Run(map[tf.Output]*tf.Tensor{
		net.Input: input,
	}, []tf.Output{net.Conv4}, nil)
	if err != nil {
		panic(err)
	}
	for _, result := range results {
		fmt.Println(result.Shape())
		//fmt.Println(result.Value().([][][][]float32))
	}
}
