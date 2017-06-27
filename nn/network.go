package nn

import (
	tf "github.com/tensorflow/tensorflow/tensorflow/go"
	"github.com/tensorflow/tensorflow/tensorflow/go/op"
	"fmt"
	"math"
)

// Neural network and func support
type Network struct {
	// Arguments for first layer of neural network
	scope         *op.Scope
	screen_height int64
	screen_width  int64
	action_size   int32

	// Layers of neural network, in tf Output type
	Input tf.Output
	Conv1 tf.Output
	Conv2 tf.Output
	Conv3 tf.Output
	Conv4 tf.Output

	// function support deep-q-learning, in tf Output type
	Qout        tf.Output
	Predict     tf.Output
	UpdateModel tf.Output

	// for compute loss
	TargetQ tf.Output
	actions tf.Output
}

func Init(scope *op.Scope, screen_height int64, screen_width int64, action_size int32) Network {
	net := new(Network)
	_one_rank1 := op.Const(scope.SubScope("conv2d"), []int32{1})
	//_zero_rank1 := op.Const(scope.SubScope("conv2d"), []int32{0})
	_one_rank0 := op.Const(scope.SubScope("conv2d"), int32(1))
	_zero_rank0 := op.Const(scope.SubScope("conv2d"), int32(0))

	// init variable for first layer
	net.scope = scope
	net.screen_width = screen_width
	net.screen_height = screen_height
	net.action_size = action_size

	// define layers of neural network
	net.Input = op.Placeholder(net.scope.SubScope("conv2d"), tf.Float,
		op.PlaceholderShape(tf.MakeShape(net.screen_height * net.screen_width)))

	shape_input := op.Const(net.scope.SubScope("conv2d"),[]int32{1, int32(net.screen_height), int32(net.screen_width), 1})

	input_reshape := op.Reshape(net.scope.SubScope("conv2d"), net.Input, shape_input)

	net.Conv1 = conv2d(net.scope, input_reshape, []int32{3, 3, 1, 32}, []int64{1,1,1,1},"conv2d")
	net.Conv2 = conv2d(net.scope, net.Conv1, []int32{3, 3, 32, 64}, []int64{1,1,1,1}, "conv2d")
	net.Conv3 = conv2d(net.scope, net.Conv2, []int32{3, 3, 64, 128}, []int64{1,1,1,1}, "conv2d")
	net.Conv4 = conv2d(net.scope, net.Conv3, []int32{3, 3, 128, net.action_size}, []int64{1,1,1,1}, "conv2d")
	// compute advantage and value for duel network
	split_dim := op.Const(net.scope.SubScope("conv2d"), int32(3))
	split := op.Split(net.scope.SubScope("conv2d"), split_dim, net.Conv4, 2)
	streamAC := split[0]
	streamVC := split[1]
	fmt.Println(streamAC.Shape())

	shape_flatten := op.Const(net.scope.SubScope("conv2d"), []int32{-1, net.action_size/2})
	streamA := op.Reshape(net.scope.SubScope("conv2d"), streamAC, shape_flatten)
	fmt.Println(streamA.Shape())

	fmt.Println(streamVC.Shape())
	streamV := op.Reshape(net.scope.SubScope("conv2d"), streamVC, shape_flatten)
	fmt.Println(streamV.Shape())


	WA := xavier_init(net.scope, []int32{net.action_size / 2, net.action_size}, true)
	WV := xavier_init(net.scope, []int32{net.action_size / 2, 1}, true)
	advantage := op.MatMul(net.scope.SubScope("conv2d"), streamA, WA)
	value := op.MatMul(net.scope.SubScope("conv2d"), streamV, WV)
	keep_dim := op.MeanKeepDims(true)
	reduce_meanA := op.Mean(net.scope.SubScope("conv2d"), advantage, _one_rank1, keep_dim)
	subtractA := op.Sub(net.scope.SubScope("conv2d"), advantage, reduce_meanA)

	Qout := op.Add(net.scope.SubScope("conv2d"), value, subtractA)
	net.Qout = Qout
	net.Predict = op.ArgMax(net.scope.SubScope("conv2d"), net.Qout, _one_rank0)

	targetQ := op.Placeholder(net.scope.SubScope("conv2d"), tf.Float, op.PlaceholderShape(tf.MakeShape()))
	actions := op.Placeholder(net.scope.SubScope("conv2d"), tf.Int32, op.PlaceholderShape(tf.MakeShape()))
	depth_action := op.Const(net.scope.SubScope("conv2d"), int32(net.action_size))
	actions_onehot := op.OneHot(net.scope.SubScope("conv2d"), actions, depth_action, _one_rank0, _zero_rank0)
	Q := op.Sum(net.scope, net.Qout, actions_onehot)
	td_err := op.Sub(net.scope.SubScope("conv2d"), targetQ, Q)
	loss := op.Square(net.scope.SubScope("conv2d"), td_err)

	// get optimize loss
	_ms := op.Const(net.scope.SubScope("conv2d"), float32(1))
	_mom := op.Const(net.scope.SubScope("conv2d"), float32(0.0))
	_lr := op.Const(net.scope.SubScope("conv2d"), float32(0.0001))
	_rho := op.Const(net.scope.SubScope("conv2d"), float32(0.9))
	_momentum := op.Const(net.scope.SubScope("conv2d"), float32(0.0))
	_esp := op.Const(net.scope.SubScope("conv2d"), float32(math.Exp(-10.0)))
	_grad := op.Const(net.scope.SubScope("conv2d"), float32(1))

	updateModel := op.ResourceApplyRMSProp(net.scope.SubScope("conv2d"), loss, _ms, _mom, _lr, _rho, _momentum, _esp, _grad, op.ResourceApplyRMSPropUseLocking(false)).Output(1)
	net.UpdateModel = updateModel

	return *net
}
