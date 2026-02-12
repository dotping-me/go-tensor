package tests

import (
	"fmt"

	"github.com/dotping-me/go-tensor/autograd"
	"github.com/dotping-me/go-tensor/nn/layers"
	"github.com/dotping-me/go-tensor/nn/losses"
	"github.com/dotping-me/go-tensor/nn/optimizers"
	"github.com/dotping-me/go-tensor/nn/topologies"
	"github.com/dotping-me/go-tensor/nn/utils"
	"github.com/dotping-me/go-tensor/tensor"
)

// Trains a transformer to estimate sentiments based on reviews
func TestSentimentAnalysis() {

	// ------------------
	//   Sample Dataset
	// ------------------

	reviews := []string{
		"good movie",     // +
		"excellent film", // +
		"bad movie",      // -
		"terrible film",  // -
	}

	// The true answers
	labels := []float32{
		0, 1, // positive
		0, 1,
		1, 0, // negative
		1, 0,
	}

	fmt.Println("Dataset:", reviews, "\nLabels :", labels)

	// ---------------------
	//   Tokenizing Inputs
	// ---------------------

	tokenizer := utils.NewTokenizer()
	tokenizer.Tokenize(reviews)
	fmt.Println("\nTokenizer Token -> ID:", tokenizer.TokenToIndexMap)
	fmt.Printf("Tokenizer ID -> Token: %v\n\n", tokenizer.IndexToTokenMap)

	inputIDs := [][]int{}
	for _, tkn := range reviews {
		ids := tokenizer.Encode(tkn)
		inputIDs = append(inputIDs, ids)
		fmt.Println("Encoding", tkn, "->", ids)
	}

	fmt.Println("\nEncoded:", inputIDs)
	fmt.Println("Shape  :", len(inputIDs), len(inputIDs[0]))

	// Flattens to 1D array
	// Also adds adding so that shapes of inputs and outputs align
	fmt.Println("\nFlattening and adding padding...")

	maxLength := 2
	padded := []float32{}

	for _, seq := range inputIDs {
		for len(seq) < maxLength {
			seq = append(seq, 0) // Padding
		}

		for _, id := range seq {
			padded = append(padded, float32(id))
		}
	}

	fmt.Println("After Padding:", padded)

	X := autograd.NewVariable(
		tensor.NewTensor([]int{len(reviews), maxLength}, padded),
		false,
	)

	// True values that will be used to calculate loss
	Y := autograd.NewVariable(
		tensor.NewTensor([]int{len(labels), 1}, labels),
		false,
	)

	// ----------------------
	//   Defining the model
	// ----------------------

	numberOfTokens := len(tokenizer.TokenToIndexMap)
	vectorSize := 8 // Hardcoded # of features to describe token
	hiddenDimension := 16

	model := topologies.NewSequential()
	model.Add(layers.NewEmbeddingLayer(numberOfTokens, vectorSize)) // NLP Inputs
	model.Add(layers.NewTransformerBlock(vectorSize, hiddenDimension))
	model.Add(layers.NewDenseLayer(8, 1)) // Outputs

	opt := optimizers.NewSGD(model.Parameters(), 0.01)

	// ----------------------
	//   Training the model
	// ----------------------

	for epoch := range 50 {
		opt.SetAllParamsGradToNil()

		// Forward pass
		outputs := model.Forward(X)
		fmt.Println(outputs.Tensor)

		// Backward pass
		loss := losses.CrossEntropy(outputs, Y)
		autograd.Backward(loss)

		// Adjusts parameters
		opt.Step()

		if epoch%5 == 0 {
			fmt.Println(
				"\nEpoch:", epoch,
				"Loss:", loss.Tensor.Data()[0],
			)
		}
	}

	// -------------
	//   Inference
	// -------------

	fmt.Println("\nStarting Inference test...")

	testReview := "good film" // Should give back positive
	testIDs := tokenizer.Encode(testReview)

	// TODO: Maybe turn this into a utility function
	// Flattens again

	// Adds padding
	for len(testIDs) < maxLength {
		testIDs = append(testIDs, 0)
	}

	// Trims too
	if len(testIDs) > maxLength {
		testIDs = testIDs[:maxLength]
	}

	testFlat := []float32{}
	for _, id := range testIDs {
		testFlat = append(testFlat, float32(id))
	}

	testVariable := autograd.NewVariable(
		tensor.NewTensor([]int{1, maxLength}, testFlat),
		false,
	)

	testOutput := model.Forward(testVariable)
	fmt.Printf("\nTest input: '%s'\nPredicted output: %v\n", testReview, testOutput.Tensor.Data())
}
