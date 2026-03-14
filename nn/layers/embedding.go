package layers

import (
	"fmt"
	"math/rand"

	"github.com/dotping-me/go-tensor/autograd"
	"github.com/dotping-me/go-tensor/tensor"
)

// The embedding layer basically converts token IDs into a Dense layer (I THINK???)
// What it does is basically describe each batch of tokens with a given set of
// features

// For example:
// [4 2] becomes [4 2 8] (using 8 features)

// TODO: Add positional encoding

type EmbeddingLayer struct {
	W              *autograd.Variable
	NumberOfTokens int // One 'row' represents one token
	VectorSize     int // How many features are used to describe a token
}

func NewEmbeddingLayer(numOfTokens, vectorSize int) *EmbeddingLayer {
	data := make([]float32, numOfTokens*vectorSize)

	for i := range data {
		data[i] = rand.Float32() * 0.01 // Small random value that will change over training
	}

	t := tensor.NewTensor([]int{numOfTokens, vectorSize}, data)
	return &EmbeddingLayer{
		W:              autograd.NewVariable(t, true),
		NumberOfTokens: numOfTokens,
		VectorSize:     vectorSize,
	}
}

// Converts token IDs to embeddings
func (e *EmbeddingLayer) Forward(x *autograd.Variable) *autograd.Variable {
	fmt.Println("\nEmbedding...")

	// NOTE: It should not change the shape of the tokens
	//       i.e [4 2] -> Embedding -> [4 2 <vectorSize>]

	ids := x.Tensor.Data()
	weights := e.W.Tensor.Data()
	data := make([]float32, len(ids)*e.VectorSize)

	for i, v := range ids {
		id := int(v) // Because Tensor holds float32 values and not integers YET!!!
		if id < 0 || id >= e.NumberOfTokens {
			id = 0 // Invalid ID -> Set to <UNK>
		}

		// Coies weights
		start := id * e.VectorSize
		end := start + e.VectorSize

		copy(data[i*e.VectorSize:(i+1)*e.VectorSize], weights[start:end])
	}

	inputShape := x.Tensor.Shape() // To avoid mutation, hopefully
	shape := append(append([]int{}, inputShape...), e.VectorSize)

	t := tensor.NewTensor(shape, data)

	// Maintains autograd
	outputVariable := autograd.NewVariable(t, true)
	outputVariable.BackwardFunc = func() {
		gradOut := outputVariable.Grad.Data()

		// During backward propagation (Gradients are nil)
		if e.W.Grad == nil {
			shape := e.W.Tensor.Shape()
			e.W.Grad = tensor.NewTensor(
				shape,
				make([]float32, tensor.NumberOfDataAsPerShape(shape)),
			)
		}

		gradW := e.W.Grad.Data()

		for i, v := range ids {
			id := int(v)

			startW := id * e.VectorSize
			startOut := i * e.VectorSize
			for j := 0; j < e.VectorSize; j++ {
				gradW[startW+j] += gradOut[startOut+j]
			}
		}
	}

	return outputVariable
}

func (e *EmbeddingLayer) Parameters() []*autograd.Variable {
	return []*autograd.Variable{e.W}
}
