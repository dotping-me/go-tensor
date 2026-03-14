package losses

import (
	"github.com/dotping-me/go-tensor/autograd"
	"github.com/dotping-me/go-tensor/tensor"
)

// Cross Entropy Loss
// CE = - ∑ y * log(softmax(x))
func CrossEntropy(logits, targets *autograd.Variable) *autograd.Variable {

	// NOTE: CSE is generally for multi-class logits, from what I read
	// 		 That's why I need a Sigmoid function for binary CSE

	return autograd.CrossEntropy(logits, targets)
}

// Cross Entropy but for binary outputs
// BCE = -1 * ( (y * lop(p)) + ((1-y) * log(1-p)) )
func BinaryCrossEntropy(logits, targets *autograd.Variable) *autograd.Variable {
	probs := autograd.Sigmoid(logits) // Calculates probabilities

	// log(p) and (1 - log(p))
	logP := autograd.Log(probs)

	one := autograd.NewVariable(tensor.NewScalar(1.0), false)
	logOneMinusP := autograd.Log(autograd.Sub(one, probs))

	// y * lop(p)
	term1 := autograd.Mul(targets, logP)

	// (1 - y) * log(1-p)
	term2 := autograd.Mul(autograd.Sub(one, targets), logOneMinusP)

	// The whole package
	// = -1 * ( Term 1 + Term 2 )

	sum := autograd.Add(term1, term2)
	loss := autograd.Neg(sum)

	// Mean all samples, for some reason
	// loss = autograd.Mean(loss, 1, false)
	// loss = autograd.Mean(loss, 0, false)

	return loss
}
