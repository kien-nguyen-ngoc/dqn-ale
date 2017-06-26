package nn

import (
	tf "github.com/tensorflow/tensorflow/tensorflow/go"
	"github.com/tensorflow/tensorflow/tensorflow/go/op"
	"math"
)

func xavier_init(scope *op.Scope, shape []int64, uniform bool) tf.Output {
	_sum := float64(shape[0] + shape[1])
	_init_range := float32(math.Sqrt(float64(6.0 / (shape[0] + shape[1]))))
	_shape := op.Placeholder(scope.SubScope("xavier"), tf.Float, op.PlaceholderShape(tf.MakeShape(1, shape[1], 1, 1)))
	_mean := op.Const(scope.SubScope("xavier"), []float32{0.0})
	_stderr := op.Const(scope.SubScope("xavier"), []float32{float32(math.Sqrt(float64(3.0 / _sum)))})
	_minval := op.Const(scope.SubScope("xavier"), []float32{-_init_range})
	_maxval := op.Const(scope.SubScope("xavier"), []float32{_init_range})

	var ret tf.Output
	if uniform {
		param := op.ParameterizedTruncatedNormal(scope.SubScope("xavier"), _shape, _mean, _stderr, _minval, _maxval, *new(op.ParameterizedTruncatedNormalAttr))
		ret = op.RandomStandardNormal(scope.SubScope("xavier"), _shape, param)
	} else {
		param := op.ParameterizedTruncatedNormal(scope.SubScope("xavier"), _shape, _mean, _stderr, _minval, _maxval, *new(op.ParameterizedTruncatedNormalAttr))
		ret = op.RandomStandardNormal(scope.SubScope("xavier"), _shape, param)
	}

	return ret
}

func conv2d(scope *op.Scope, input tf.Output, out_size int64, kernel_size []int64, scope_name string) tf.Output {
	//input := op.Placeholder(net.scope.SubScope("input"), tf.Float, op.PlaceholderShape(tf.MakeShape(1, net.screen_height, net.screen_width, 3)))
	//fmt.Println(input.Op.Name())

	kernel := xavier_init(scope.SubScope(scope_name), kernel_size, false)

	conv := op.Conv2D(scope.SubScope(scope_name), input, kernel, []int64{1, 1, 1, 1}, "VALID")

	//biases := op.Const(net.scope.SubScope("kernel"), [3][3]int32{{1,1,1},{1,1,1},{1,1,1}})
	//biases := tf.Variable(tf.constant(0.0, shape=[128], dtype=tf.float32), trainable=True, name="biases")

	//bias := op.BiasAdd(net.scope.SubScope("bias"), conv, biases)
	conv2d := op.Tanh(scope.SubScope(scope_name), conv)
	return conv2d
}
