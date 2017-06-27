package nn

import (
	tf "github.com/tensorflow/tensorflow/tensorflow/go"
	"github.com/tensorflow/tensorflow/tensorflow/go/op"
	"math"
)

func xavier_init(scope *op.Scope, shape []int32, uniform bool) tf.Output {
	_zero := op.Const(scope.SubScope("xavier"), []float32{0.0})
	_one := op.Const(scope.SubScope("xavier"), []float32{1.0})

	_minus_one := op.Const(scope.SubScope("xavier"), []float32{-1.0})

	_sum := float64(shape[0] + shape[1])
	_range := float32(math.Sqrt(float64(6.0 / _sum)))
	_shape := op.Const(scope.SubScope("xavier"), shape)
	_stderr := op.Const(scope.SubScope("xavier"), []float32{float32(math.Sqrt(float64(3.0 / _sum)))})
	_minvalue := op.Const(scope.SubScope("xavier"), []float32{-_range})
	_maxvalue := op.Const(scope.SubScope("xavier"), []float32{_range})

	var ret tf.Output
	if uniform {
		ret = op.ParameterizedTruncatedNormal(scope.SubScope("xavier"), _shape,
			_zero, _one, _minvalue, _maxvalue)
	} else {
		ret = op.ParameterizedTruncatedNormal(scope.SubScope("xavier"), _shape,
			_zero, _stderr, _minus_one, _one)
	}

	return ret
}

func conv2d(scope *op.Scope, input tf.Output, filter []int32, strides []int64, scope_name string) tf.Output {

	kernel := xavier_init(scope.SubScope(scope_name), filter, false)

	conv := op.Conv2D(scope.SubScope(scope_name), input, kernel, strides, "VALID")

	//biases := op.Const(net.scope.SubScope("kernel"), [3][3]int32{{1,1,1},{1,1,1},{1,1,1}})
	//biases := tf.Variable(tf.constant(0.0, shape=[128], dtype=tf.float32), trainable=True, name="biases")

	//bias := op.BiasAdd(net.scope.SubScope("bias"), conv, biases)
	conv2d := op.Tanh(scope.SubScope(scope_name), conv)
	return conv2d
}
