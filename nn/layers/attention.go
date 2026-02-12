package layers

import (
	"math"
	"math/rand"

	"github.com/dotping-me/go-tensor/autograd"
	"github.com/dotping-me/go-tensor/tensor"
)

type AttentionLayer struct {
	Wq, Wk, Wv *autograd.Variable // These are basically weights
	Dk         float32
}

func NewAttentionLayer(vectorSize int) *AttentionLayer {
	numberOfData := vectorSize * vectorSize
	data := make([]float32, numberOfData)

	for i := range numberOfData {
		data[i] = rand.Float32() * 0.01
	}

	q := tensor.NewTensor([]int{vectorSize, vectorSize}, data)
	k := tensor.NewTensor([]int{vectorSize, vectorSize}, data)
	v := tensor.NewTensor([]int{vectorSize, vectorSize}, data)

	return &AttentionLayer{
		Wq: autograd.NewVariable(q, true),
		Wk: autograd.NewVariable(k, true),
		Wv: autograd.NewVariable(v, true),
		Dk: float32(vectorSize),
	}
}

func (a *AttentionLayer) Forward(x *autograd.Variable) *autograd.Variable {

	// Formula:
	// Attention scores (how each word relates to every other word)
	// = Softmax(Q * (K^t) / Sqrt(Dk)) * V

	// But Q = X * Wq
	//     K = X * Wk
	//     V = X * Wv

	Q := autograd.Matrix2dMul(x, a.Wq)
	K := autograd.Matrix2dMul(x, a.Wk)
	V := autograd.Matrix2dMul(x, a.Wv)

	// Calculates attention scores = Q * (K^t)
	KtTensor, _ := K.Tensor.Transpose2D()
	Kt := autograd.NewVariable(KtTensor, false)
	scores := autograd.Matrix2dMul(Q, Kt)

	// Divide (Scale) by Sqrt(Dk)
	scaleValue := float32(math.Sqrt(float64(a.Dk)))
	scaleTensor := tensor.NewScalar(scaleValue)
	scaleVariable := autograd.NewVariable(scaleTensor, false)

	scaled := autograd.Div(scores, scaleVariable)

	// (Softmax of Q * (K^t)) * V
	attnWeights := autograd.Softmax(scaled, 1)
	outputVariable := autograd.Matrix2dMul(attnWeights, V)

	return outputVariable
}

func (a *AttentionLayer) Parameters() []*autograd.Variable {
	return []*autograd.Variable{a.Wq, a.Wk, a.Wv}
}
