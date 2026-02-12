package losses

import "github.com/dotping-me/go-tensor/autograd"

// Cross Entropy Loss
// CE = - ∑ y * log(softmax(x))
func CrossEntropy(logits, targets *autograd.Variable) *autograd.Variable {

	// = log(softmax(x))
	probs := autograd.Softmax(logits, 1)
	logProbs := autograd.Log(probs)

	// = y * log(softmax(x))
	masked := autograd.Mul(logProbs, targets)

	sum := autograd.Sum(masked, 1, false) // ∑ y * log(softmax(x))
	loss := autograd.Neg(sum)             // Just inverts signs

	return loss
}
