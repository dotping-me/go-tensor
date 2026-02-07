package nn

import "github.com/dotping-me/go-tensor/autograd"

// This is just basically a Layer but it's here so users can just call Model instead
// of Layer which makes way more sense
type Model interface {
	Forward(v *autograd.Variable) *autograd.Variable
	Parameters() []*autograd.Variable
}

type Layer interface {
	Forward(v *autograd.Variable) *autograd.Variable
	Parameters() []*autograd.Variable
}

type Sequential struct { // Sequention model which plugs into Model interface
	layers []Layer
}

func NewSequential() *Sequential {
	return &Sequential{}
}

func (s *Sequential) Add(l Layer) {
	s.layers = append(s.layers, l)
}

// Calls Forward on every Layer of the sequential model
func (s *Sequential) Forward(v *autograd.Variable) *autograd.Variable {

	// Accumulates outputs
	allOutputs := v
	for _, l := range s.layers {
		allOutputs = l.Forward(allOutputs)
	}

	return allOutputs
}

// Collects all the variables from all layers iteratively
func (s *Sequential) Parameters() []*autograd.Variable {
	allParams := []*autograd.Variable{}

	for _, l := range s.layers {
		allParams = append(allParams, l.Parameters()...)
	}

	return allParams
}
