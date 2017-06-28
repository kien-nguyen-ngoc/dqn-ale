package nn

import (
	tf "github.com/tensorflow/tensorflow/tensorflow/go"
	"github.com/tensorflow/tensorflow/tensorflow/go/op"
	"fmt"
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

	// for test only
	Test tf.Output
}

func Init(scope *op.Scope, screen_height int64, screen_width int64, action_size int32) Network {
	last_layer := int32(512)
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

	// process frame through four convolutional layers,
	// must reduce shape at last layer to [1,1,1,last_layer_size]
	net.Conv1 = conv2d(net.scope, input_reshape, []int32{3, 3, 1, 32}, []int64{1,1,1,1},"conv2d")
	net.Conv2 = conv2d(net.scope, net.Conv1, []int32{3, 3, 32, 64}, []int64{1,1,1,1}, "conv2d")
	net.Conv3 = conv2d(net.scope, net.Conv2, []int32{3, 3, 64, 128}, []int64{1,1,1,1}, "conv2d")
	net.Conv4 = conv2d(net.scope, net.Conv3, []int32{3, 3, 128, last_layer}, []int64{1,1,1,1}, "conv2d")

	// compute advantage and value for duel network
	split_dim := op.Const(net.scope.SubScope("conv2d"), int32(3))
	split := op.Split(net.scope.SubScope("conv2d"), split_dim, net.Conv4, 2)
	streamAC := split[0]
	streamVC := split[1]

	shape_flatten := op.Const(net.scope.SubScope("conv2d"), []int32{-1, last_layer / 2})
	streamA := op.Reshape(net.scope.SubScope("conv2d"), streamAC, shape_flatten)
	streamV := op.Reshape(net.scope.SubScope("conv2d"), streamVC, shape_flatten)


	WA := xavier_init(net.scope, []int32{last_layer / 2, net.action_size}, true)
	WV := xavier_init(net.scope, []int32{last_layer / 2, 1}, true)
	advantage := op.MatMul(net.scope.SubScope("conv2d"), streamA, WA)
	value := op.MatMul(net.scope.SubScope("conv2d"), streamV, WV)
	keep_dim := op.MeanKeepDims(true)
	reduce_meanA := op.Mean(net.scope.SubScope("conv2d"), advantage, _one_rank1, keep_dim)
	subtractA := op.Sub(net.scope.SubScope("conv2d"), advantage, reduce_meanA)

	Qout := op.Add(net.scope.SubScope("conv2d"), value, subtractA)
	net.Qout = Qout

	net.Predict = op.ArgMax(net.scope.SubScope("conv2d"), net.Qout, _one_rank0)


	targetQ := op.Placeholder(net.scope.SubScope("conv2d"), tf.Float)
	fmt.Println(targetQ.Shape())
	actions := op.Placeholder(net.scope.SubScope("conv2d"), tf.Int32)
	depth_action := op.Const(net.scope.SubScope("conv2d"), int32(net.action_size))
	actions_onehot := op.OneHot(net.scope.SubScope("conv2d"), actions, depth_action, _one_rank0, _zero_rank0)
	Q := op.Sum(net.scope, net.Qout, actions_onehot)
	td_err := op.Sub(net.scope.SubScope("conv2d"), targetQ, Q)
	loss := op.Square(net.scope.SubScope("conv2d"), td_err)
	fmt.Println(Qout.Shape())
	fmt.Println(actions_onehot.Shape())
	fmt.Println(loss.Shape())

	// get optimize loss
	//_var_ := op.VarHandleOp(net.scope.SubScope("conv2d"), tf.Float, loss.Shape())
	_ms_ := op.VarHandleOp(net.scope.SubScope("conv2d"), tf.Float, tf.ScalarShape())
	_mom_ := op.VarHandleOp(net.scope.SubScope("conv2d"), tf.Float, tf.ScalarShape())
	_ms := op.Const(net.scope.SubScope("conv2d"), float32(0.0))
	_mom := op.Const(net.scope.SubScope("conv2d"), float32(0.0))
	//_var := op.AssignVariableOp(net.scope.SubScope("conv2d"), _var_, loss)
	_m := op.AssignVariableOp(net.scope.SubScope("conv2d"), _ms_, _ms)
	_v := op.AssignVariableOp(net.scope.SubScope("conv2d"), _mom_, _mom)

	out := tf.Output{_m,0}

	//fmt.Println(_var_)
	//fmt.Println(_var.NumOutputs())
	net.Test = op.VarIsInitializedOp(net.scope.SubScope("conv2d"), out)
	//fmt.Println(net.Test.Shape())
	fmt.Println(_m)
	fmt.Println(_v)

	//fmt.Println(_m.Shape())
	//fmt.Println(_v.Shape())
	//
	//
	//_lr := op.Const(net.scope.SubScope("conv2d"), float32(0.0001))
	//_beta1 := op.Const(net.scope.SubScope("conv2d"), float32(0.9))
	//_beta2 := op.Const(net.scope.SubScope("conv2d"), float32(0.999))
	//_beta1_power := op.Const(net.scope.SubScope("conv2d"), float32(1.0))
	//_beta2_power := op.Const(net.scope.SubScope("conv2d"), float32(1.0))
	//_esp := op.Const(net.scope.SubScope("conv2d"), float32(math.Exp(-8.0)))
	//_grad := op.Const(net.scope.SubScope("conv2d"), float32(1))
	//
	//
	//updateModel := op.ResourceApplyAdam(net.scope.SubScope("conv2d"),
	//	_var, _m, _v, _beta1_power, _beta2_power, _lr, _beta1, _beta2, _esp, _grad,
	//	op.ResourceApplyAdamUseLocking(false)).Output(0)
	//
	//net.UpdateModel = updateModel

	return *net
}
