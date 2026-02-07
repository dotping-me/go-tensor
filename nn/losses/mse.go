package losses

import "github.com/dotping-me/go-tensor/autograd"

// Mean Squared Error Loss function
// MSE(y, Y) = (1/N) * ∑ ((y - Y) ** 2)
func MeanSquaredError(predictedY, trueY *autograd.Variable) *autograd.Variable {
	diff := autograd.Sub(predictedY, trueY) // (y - Y)
	sq := autograd.Mul(diff, diff)          // (y - Y) ** 2
	loss := autograd.Sum(sq, 0, false)      // ∑ ((y - Y) ** 2)

	return loss
}
