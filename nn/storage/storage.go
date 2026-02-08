package storage

import (
	"encoding/json"
	"os"

	"github.com/dotping-me/go-tensor/autograd"
	"github.com/dotping-me/go-tensor/tensor"
)

func SaveModel(fpath string, params []*autograd.Variable) error {
	f, err := os.Create(fpath)
	if err != nil {
		return err
	}
	defer f.Close()

	// Saves parameters as JSON
	jsonObj := make([]map[string]interface{}, len(params))
	for i, p := range params {
		jsonObj[i] = map[string]interface{}{
			"shape": p.Tensor.Shape(),
			"data":  p.Tensor.Data(),
		}
	}

	return json.NewEncoder(f).Encode(jsonObj)
}

func LoadModel(fpath string) ([]*autograd.Variable, error) {
	f, err := os.Open(fpath)
	if err != nil {
		return nil, err
	}
	defer f.Close()

	// Decodes JSOn
	var jsonObj []map[string]interface{}
	if err := json.NewDecoder(f).Decode(&jsonObj); err != nil {
		return nil, err
	}

	// Turns plain values into a tensor and then into an autograd variable
	params := make([]*autograd.Variable, len(jsonObj))
	for i, jsonData := range jsonObj {
		shape := jsonData["shape"].([]interface{}) // Cast values to an interface
		outputShape := make([]int, len(shape))
		for j, val := range shape {
			outputShape[j] = int(val.(float64))
		}

		data := jsonData["data"].([]interface{}) // Cast values to an interface
		outputData := make([]float32, len(data))
		for j, val := range data {
			outputData[j] = float32(val.(float64))
		}

		outputTensor := tensor.NewTensor(outputShape, outputData)
		params[i] = autograd.NewVariable(outputTensor, true)
	}

	return params, nil
}
