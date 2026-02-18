package layers

import (
	"fmt"
	"math/rand"

	"github.com/dotping-me/go-tensor/autograd"
	"github.com/dotping-me/go-tensor/tensor"
)

// The embedding layer basically converts token IDs into a Dense layer

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
	numOfTokens := len(ids)
	data := make([]float32, numOfTokens*e.VectorSize)

	for i, v := range ids {
		id := int(v) // Because Tensor holds float32 values and not integers YET!!!
		if id < 0 || id >= e.NumberOfTokens {
			id = 0 // Invalid ID -> Set to <UNK>
		}

		// Pastes the weights
		for j := 0; j < e.VectorSize; j++ {
			data[i*e.VectorSize+j] = e.W.Tensor.Data()[id*e.VectorSize+j]
		}
	}

	t := tensor.NewTensor([]int{numOfTokens, e.VectorSize}, data)
	return autograd.NewVariable(t, false)
}

func (e *EmbeddingLayer) Parameters() []*autograd.Variable {
	return []*autograd.Variable{e.W}
}
