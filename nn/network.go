package nn

import (
	tf "github.com/tensorflow/tensorflow/tensorflow/go"
	"github.com/tensorflow/tensorflow/tensorflow/go/op"
	"math"
)

// Neural network and func support
type Network struct {
	// Arguments for first layer of neural network
	scope           *op.Scope
	screen_height   int64
	screen_width    int64
	action_size     int32
	last_layer_size int32

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
	Actions tf.Output

	// for init tf.Variable
	Init_var *tf.Operation
	Init_m   *tf.Operation
	Init_v   *tf.Operation

	// for test only
	Test tf.Output
}

func Init(scope *op.Scope, screen_height int64, screen_width int64, action_size int32, last_layer_size int32) Network {
	net := new(Network)
	_one_rank1 := op.Const(scope.SubScope("conv2d_one_rank1"), []int32{1})
	//_zero_rank1 := op.Const(scope.SubScope("conv2d"), []int32{0})
	_one_rank0 := op.Const(scope.SubScope("conv2d_one_rank0"), int32(1))
	_zero_rank0 := op.Const(scope.SubScope("conv2d_zero_rank0"), int32(0))

	// init variable for first layer
	net.scope = scope
	net.screen_width = screen_width
	net.screen_height = screen_height
	net.action_size = action_size
	net.last_layer_size = last_layer_size

	// define layers of neural network
	net.Input = op.Placeholder(net.scope.SubScope("conv2d_placeholder_input"), tf.Float,
		op.PlaceholderShape(tf.MakeShape(net.screen_height*net.screen_width)))

	shape_input := op.Const(net.scope.SubScope("conv2d_shape_input"), []int32{1, int32(net.screen_height), int32(net.screen_width), 1})

	input_reshape := op.Reshape(net.scope.SubScope("conv2d_input_reshape"), net.Input, shape_input)

	// process frame through four convolutional layers,
	// must reduce shape at last layer to [1,1,1,last_layer_size]
	net.Conv1 = conv2d(net.scope, input_reshape, []int32{9, 9, 1, 32}, []int64{1, 2, 2, 1}, "conv2d_conv1")
	net.Conv2 = conv2d(net.scope, net.Conv1, []int32{5, 5, 32, 64}, []int64{1, 2, 2, 1}, "conv2d_conv2")
	net.Conv3 = conv2d(net.scope, net.Conv2, []int32{5, 5, 64, 128}, []int64{1, 2, 2, 1}, "conv2d_conv3")
	net.Conv4 = conv2d(net.scope, net.Conv3, []int32{3, 3, 128, net.last_layer_size}, []int64{1, 2, 2, 1}, "conv2d_conv4")

	// compute advantage and value for duel network
	split_dim := op.Const(net.scope.SubScope("conv2d_split_dim"), int32(3))
	split := op.Split(net.scope.SubScope("conv2d_split"), split_dim, net.Conv4, 2)
	streamAC := split[0]
	streamVC := split[1]

	shape_flatten := op.Const(net.scope.SubScope("conv2d_shape_flatten"), []int32{-1, net.last_layer_size / 2})
	streamA := op.Reshape(net.scope.SubScope("conv2d_stream_a"), streamAC, shape_flatten)
	streamV := op.Reshape(net.scope.SubScope("conv2d_stream_v"), streamVC, shape_flatten)

	WA := xavier_init(net.scope, []int32{net.last_layer_size / 2, net.action_size}, true)
	WV := xavier_init(net.scope, []int32{net.last_layer_size / 2, 1}, true)
	advantage := op.MatMul(net.scope.SubScope("conv2d_advantage"), streamA, WA)
	value := op.MatMul(net.scope.SubScope("conv2d_value"), streamV, WV)
	keep_dim := op.MeanKeepDims(true)
	reduce_meanA := op.Mean(net.scope.SubScope("conv2d_reduce_mean_a"), advantage, _one_rank1, keep_dim)
	subtractA := op.Sub(net.scope.SubScope("conv2d_subtract_a"), advantage, reduce_meanA)

	Qout := op.Add(net.scope.SubScope("conv2d_q_out"), value, subtractA)
	net.Qout = Qout

	net.Predict = op.ArgMax(net.scope.SubScope("conv2d_predict"), net.Qout, _one_rank0)

	targetQ := op.Placeholder(net.scope.SubScope("conv2d_target_q"), tf.Float)
	actions := op.Placeholder(net.scope.SubScope("conv2d_actions"), tf.Int32)
	depth_action := op.Const(net.scope.SubScope("conv2d_depth_action"), int32(net.action_size))
	actions_onehot := op.OneHot(net.scope.SubScope("conv2d_actions_onthot"), actions, depth_action, _one_rank0, _zero_rank0)
	Q := op.Sum(net.scope, net.Qout, actions_onehot)
	td_err := op.Sub(net.scope.SubScope("conv2d_td_err"), targetQ, Q)
	loss := op.Square(net.scope.SubScope("conv2d_loss"), td_err)

	net.TargetQ = targetQ
	net.Actions = actions

	// get optimize loss
	_var := op.VarHandleOp(net.scope.SubScope("conv2d_var"), tf.Float, loss.Shape())
	_m := op.VarHandleOp(net.scope.SubScope("conv2d_m"), tf.Float, tf.ScalarShape())
	_v := op.VarHandleOp(net.scope.SubScope("conv2d_v"), tf.Float, tf.ScalarShape())
	_zero_float32_rank0 := op.Const(net.scope.SubScope("conv2d_zero_float32_rank0"), float32(0.0))

	net.Init_var = op.AssignVariableOp(net.scope.SubScope("conv2d_init_var"), _var, loss)
	net.Init_m = op.AssignVariableOp(net.scope.SubScope("conv2d_init_m"), _m, _zero_float32_rank0)
	net.Init_v = op.AssignVariableOp(net.scope.SubScope("conv2d_init_v"), _v, _zero_float32_rank0)

	net.Test = op.VarIsInitializedOp(scope.SubScope("conv2d_check_init_variable"), _m)
	//fmt.Println(net.Test.Shape())

	//
	//
	_lr := op.Const(net.scope.SubScope("conv2d"), float32(0.0001))
	_beta1 := op.Const(net.scope.SubScope("conv2d"), float32(0.9))
	_beta2 := op.Const(net.scope.SubScope("conv2d"), float32(0.999))
	_beta1_power := op.Const(net.scope.SubScope("conv2d"), float32(1.0))
	_beta2_power := op.Const(net.scope.SubScope("conv2d"), float32(1.0))
	_esp := op.Const(net.scope.SubScope("conv2d"), float32(math.Exp(-8.0)))
	_grad := op.Const(net.scope.SubScope("conv2d"), float32(1))
	//
	//
	updateModel := op.ResourceApplyAdam(net.scope.SubScope("conv2d"),
		_var, _m, _v, _beta1_power, _beta2_power, _lr, _beta1, _beta2, _esp, _grad,
		op.ResourceApplyAdamUseLocking(false)).Output(0)

	net.UpdateModel = updateModel

	return *net
}
