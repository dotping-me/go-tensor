package optimizers

import (
	"github.com/dotping-me/go-tensor/autograd"
	"github.com/dotping-me/go-tensor/tensor"
)

// Stochastic Gradient Descent
type SGD struct {
	Params       []*autograd.Variable
	LearningRate float32
}

func NewSGD(params []*autograd.Variable, learningRate float32) *SGD {
	return &SGD{Params: params, LearningRate: learningRate}
}

// Basically it updates the parameters, adjusting them by learningRate each time
// to make them converge and reduce loss
func (s *SGD) Step() {
	lrScalar := tensor.NewScalar(s.LearningRate)

	// Formula:
	// param = param - learningRate * gradParam

	for _, p := range s.Params {
		if p.Grad != nil {
			update, _ := p.Grad.Mul(lrScalar)
			p.Tensor, _ = p.Tensor.Sub(update)
		}
	}
}

func (s *SGD) SetAllParamsGradToNil() {
	for _, p := range s.Params {
		p.SetGradToNil()
	}
}
