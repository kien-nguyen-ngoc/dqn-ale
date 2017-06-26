package nn

import (
	tf "github.com/tensorflow/tensorflow/tensorflow/go"
	"github.com/tensorflow/tensorflow/tensorflow/go/op"
	"math"
)

// Neural network and func support
type Network struct {
	// Arguments for first layer of neural network
	scope         *op.Scope
	screen_height int64
	screen_width  int64
	action_size   int64

	// Layers of neural network, in tf Output type
	input tf.Output
	conv1 tf.Output
	conv2 tf.Output
	conv3 tf.Output
	conv4 tf.Output

	// function support deep-q-learning, in tf Output type
	Qout tf.Output
	predict tf.Output
	updateModel tf.Output

	// for compute loss
	targetQ tf.Output
	actions tf.Output
}

func _init(scope *op.Scope, screen_height int64, screen_width int64, action_size int64) Network {
	net := new(Network)
	one := op.Const(scope.SubScope("conv2d"), []float32{1})
	zero:= op.Const(scope.SubScope("conv2d"), []float32{0})

	// init variable for first layer
	net.scope = scope
	net.screen_width = screen_width
	net.screen_height = screen_height
	net.action_size = action_size

	// define layers of neural network
	input := op.Placeholder(scope.SubScope("conv2d"), tf.Float, op.PlaceholderShape(tf.MakeShape(1, 3, 3, 1)))
	net.input = input
	net.conv1 = conv2d(net.scope, net.input, 32, []int64{screen_width, screen_height}, "conv2d")
	net.conv2 = conv2d(net.scope, net.conv1, 64, []int64{screen_width, screen_height}, "conv2d")
	net.conv3 = conv2d(net.scope, net.conv2, 64, []int64{screen_width, screen_height}, "conv2d")
	net.conv4 = conv2d(net.scope, net.conv3, action_size, []int64{screen_width, screen_height}, "conv2d")

	// compute advantage and value for duel network
	split_dim := op.Const(scope.SubScope("conv2d"), []float32{3})
	split := op.Split(scope, split_dim, net.conv4, 2)
	streamA := split[0]
	streamV := split[1]
	WA := xavier_init(net.scope, []int64{net.action_size / 2, net.action_size}, true)
	WV := xavier_init(net.scope, []int64{net.action_size / 2, 1}, true)
	advantage := op.MatMul(net.scope, streamA, WA)
	value := op.MatMul(net.scope, streamV, WV)
	keep_dim := op.MeanKeepDims(true)
	reduce_meanA := op.Mean(net.scope, advantage, one, keep_dim)
	subtractA := op.Sub(net.scope, advantage, reduce_meanA)

	Qout := op.Add(net.scope, value, subtractA)
	net.Qout = Qout
	net.predict = op.ArgMax(net.scope, net.Qout, one)

	targetQ := op.Placeholder(scope.SubScope("conv2d"), tf.Float, op.PlaceholderShape(tf.MakeShape()))
	actions := op.Placeholder(scope.SubScope("conv2d"), tf.Int32, op.PlaceholderShape(tf.MakeShape()))
	depth_action := op.Const(scope.SubScope("conv2d"), []int64{net.action_size})
	actions_onehot := op.OneHot(net.scope, actions, depth_action, one, zero)
	Q := op.Sum(net.scope, net.Qout, actions_onehot)
	td_err := op.Sub(net.scope, targetQ, Q)
	loss := op.Square(net.scope, td_err)

	// get optimize loss
	_ms := op.Const(scope.SubScope("conv2d"), []float32{1})
	_mom := op.Const(scope.SubScope("conv2d"), []float32{0.0})
	_lr := op.Const(scope.SubScope("conv2d"), []float32{0.0001})
	_rho := op.Const(scope.SubScope("conv2d"), []float32{0.9})
	_momentum := op.Const(scope.SubScope("conv2d"), []float32{0.0})
	_esp := op.Const(scope.SubScope("conv2d"), []float32{float32(math.Exp(-10.0))})
	_grad := op.Const(scope.SubScope("conv2d"), []float32{1})

	updateModel := op.ResourceApplyRMSProp(net.scope, loss, _ms, _mom, _lr, _rho, _momentum, _esp, _grad, op.ResourceApplyRMSPropUseLocking(false)).Output(1)
	net.updateModel = updateModel

	return *net
}
