// TODO: Use more meaningful variable names or at least break stuff down

package tensor

// Softmax function (It looks like a fever dream)
// It basically maps outputs (a tensor) onto a normal distribution
func (x *Tensor) Softmax(axisIndex int) (*Tensor, error) {

	// Formula:
	// softmax(x_i) = e**x_i / (∑_j (e**x_i))

	// Which basically is, at least I think:
	// = (e ** (x - LogSumExp(x))) / ∑ (e ** (x - LogSumExp(x)))

	lse, err := x.LogSumExp(axisIndex, true)
	if err != nil {
		return nil, err
	}

	diff, err := x.Sub(lse)
	if err != nil {
		return nil, err
	}

	exps, err := diff.Exp()
	if err != nil {
		return nil, err
	}

	sum, err := exps.Sum(axisIndex, true)
	if err != nil {
		return nil, err
	}

	return exps.Div(sum)
}

// ------------------
//   Loss Functions
// ------------------

// Basically (I'm using basically a lot because I do not have it in me to explain
// it properly) calculates how wrong the outputs were

// Mean Squared Error Loss function
func MeanSquaredError(actualY, expectedY *Tensor, axisIndex int) (*Tensor, error) {

	// Formula:
	// MSE(y, Y) = (1/N) * ∑ ((y - Y) ** 2)

	// (y - Y)
	diff, err := actualY.Sub(expectedY)
	if err != nil {
		return nil, err
	}

	// (y - Y) ** 2
	squared, err := diff.Pow(2)
	if err != nil {
		return nil, err
	}

	// ∑ ((y - Y) ** 2)
	sum, err := squared.Sum(axisIndex, false)
	if err != nil {
		return nil, err
	}

	// (1/N) * ∑ ((y - Y) ** 2)
	N := float32(actualY.shape[axisIndex])
	nT := NewTensor([]int{1}, []float32{N})

	return sum.Div(nT)
}

// Cross-Entropy Loss function
func CrossEntropy(x, y *Tensor, axisIndex int) (*Tensor, error) {

	// Formula:
	// L = - ∑ ( y * log( softmax(x) ))

	// So basically:
	// = -1 * ( y.Mul(x.Softmax().Log()) ).Sum()

	probs, err := x.Softmax(axisIndex)
	if err != nil {
		panic(err)
	}

	logProbs, err := probs.Log()
	if err != nil {
		panic(err)
	}

	masked, err := y.Mul(logProbs)
	if err != nil {
		panic(err)
	}

	// Sumation part
	loss, err := masked.Sum(axisIndex, false)
	if err != nil {
		panic(err)
	}

	return loss.Neg()
}
