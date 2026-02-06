package tests

import (
	"fmt"

	"github.com/dotping-me/go-tensor/internal/autograd"
	"github.com/dotping-me/go-tensor/internal/tensor"
)

func TestScalarBackward() {
	fmt.Println("\n\n----- Test Backward Reg #1 -----")
	fmt.Println("Doing y = (x * x) + x")
	fmt.Println("-> dy/dx = 2x + 1 which at x = 3, dy/dx = 7")

	xTensor := tensor.NewTensor([]int{1}, []float32{3.0})
	x := autograd.NewVariable(xTensor, true)
	y := autograd.Add(autograd.Mul(x, x), x)

	autograd.Backward(y) // Computes gradients

	// Should be 7
	fmt.Printf("\nAt x = %.2f, dy/dx = %.2f\n", x.Tensor.Data()[0], x.Grad.Data()[0])
}

func TestMatMulBackward() {
	X := autograd.NewVariable(
		tensor.NewTensor([]int{2, 2}, []float32{
			1, 2,
			3, 4,
		}),
		false,
	)

	W := autograd.NewVariable(
		tensor.NewTensor([]int{2, 1}, []float32{
			0.5,
			-1.0,
		}),
		true,
	)

	Y := autograd.Matrix2dMul(X, W)
	L := autograd.Sum(Y, 0, false)

	autograd.Backward(L)

	fmt.Println("Grad W:", W.Grad)
}

func NumericalGrad(f func() float32, w *tensor.Tensor, eps float32) float32 {
	orig := w.Data()[0]

	w.Data()[0] = orig + eps
	p := f()

	w.Data()[0] = orig - eps
	m := f()

	w.Data()[0] = orig
	return (p - m) / (2 * eps)
}

func TestNumericGrad() {
	W := tensor.NewTensor([]int{1}, []float32{2.0})
	w := autograd.NewVariable(W, true)

	f := func() float32 {
		y := autograd.Mul(w, w)
		return y.Tensor.Data()[0]
	}

	y := autograd.Mul(w, w)
	autograd.Backward(y)

	analytic := w.Grad.Data()[0]
	numeric := NumericalGrad(f, W, 1e-3)

	fmt.Println("Analytic:", analytic)
	fmt.Println("Numeric:", numeric)
}
