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
	dataQ := make([]float32, numberOfData)
	dataK := make([]float32, numberOfData)
	dataV := make([]float32, numberOfData)

	for i := range numberOfData {
		dataQ[i] = rand.Float32() * 0.01
		dataK[i] = rand.Float32() * 0.01
		dataV[i] = rand.Float32() * 0.01
	}

	q := tensor.NewTensor([]int{vectorSize, vectorSize}, dataQ)
	k := tensor.NewTensor([]int{vectorSize, vectorSize}, dataK)
	v := tensor.NewTensor([]int{vectorSize, vectorSize}, dataV)

	return &AttentionLayer{
		Wq: autograd.NewVariable(q, true),
		Wk: autograd.NewVariable(k, true),
		Wv: autograd.NewVariable(v, true),
		Dk: float32(vectorSize),
	}
}

// Calculates attention scores
func (a *AttentionLayer) Forward(x *autograd.Variable) *autograd.Variable {

	// NOTE: Assuming that input is of rank >= 3
	shape := x.Tensor.Shape()
	B, S, E := shape[0], shape[1], shape[2]

	xFlat := x.Reshape([]int{B * S, E})

	// Linear projections ([B, S, E] to [B*S, E])
	Qflat := autograd.Matrix2dMul(xFlat, a.Wq)
	Kflat := autograd.Matrix2dMul(xFlat, a.Wk)
	Vflat := autograd.Matrix2dMul(xFlat, a.Wv)

	// Reshape back to [B, S, E], because that's what we're gonna use
	Q := Qflat.Reshape([]int{B, S, E})
	K := Kflat.Reshape([]int{B, S, E})
	V := Vflat.Reshape([]int{B, S, E})

	outputData := make([]float32, B*S*E)
	for b := range B {

		// Get batches (= B of [S, E])
		qBatch, _ := Q.Tensor.Get2DSliceAtParentIndex([]int{b})
		kBatch, _ := K.Tensor.Get2DSliceAtParentIndex([]int{b})
		vBatch, _ := V.Tensor.Get2DSliceAtParentIndex([]int{b})

		qVar := autograd.NewVariable(qBatch, x.RequiresGrad)
		kVar := autograd.NewVariable(kBatch, x.RequiresGrad)
		vVar := autograd.NewVariable(vBatch, x.RequiresGrad)

		// Calculates attention scores
		// [S, S] = Q * K^T
		KtTensor, _ := kVar.Tensor.Transpose2D()
		Kt := autograd.NewVariable(KtTensor, false)
		scores := autograd.Matrix2dMul(qVar, Kt)

		// Scale by sqrt(Dk)
		scale := float32(math.Sqrt(float64(a.Dk)))
		scaleVar := autograd.NewVariable(tensor.NewScalar(scale), false)
		scaled := autograd.Div(scores, scaleVar)

		attnWeights := autograd.Softmax(scaled, 1)            // Softmax along sequence axis
		attnOutput := autograd.Matrix2dMul(attnWeights, vVar) // Multiply by weights

		// Compiles everything
		for s := range S {
			for e := range E {
				outputData[(b*S+s)*E+e] = attnOutput.Tensor.Data()[s*E+e]
			}
		}
	}

	outTensor := tensor.NewTensor([]int{B, S, E}, outputData)
	return autograd.NewVariable(outTensor, x.RequiresGrad)
}
func (a *AttentionLayer) Parameters() []*autograd.Variable {
	return []*autograd.Variable{a.Wq, a.Wk, a.Wv}
}
