package utils

import "strings"

type Tokenizer struct {
	TokenToIndexMap map[string]int
	IndexToTokenMap map[int]string
}

// Stores 'same' data twice to increase lookup, maybe there are some other faster
// ways to do it than this
func NewTokenizer() *Tokenizer {
	t := &Tokenizer{
		TokenToIndexMap: make(map[string]int),
		IndexToTokenMap: make(map[int]string),
	}

	// Reserves a memory location for special token
	t.TokenToIndexMap["<UNK>"] = 0
	t.IndexToTokenMap[0] = "<UNK>"

	return t
}

// Loops through an array of strings and appends unique tokens to the struct arrays
func (t *Tokenizer) Tokenize(texts []string) {
	id := 1
	for _, txt := range texts {
		for _, token := range strings.Fields(txt) {
			if _, exists := t.TokenToIndexMap[token]; !exists {
				t.TokenToIndexMap[token] = id
				t.IndexToTokenMap[id] = token
				id++
			}
		}
	}
}

// Iterates over a string and returns an array of indices corresponding to the
// respective tokens in that string
func (t *Tokenizer) Encode(text string) []int {
	tokens := strings.Fields(text)
	ids := make([]int, len(tokens))

	for i, tkn := range tokens {
		if id, ok := t.TokenToIndexMap[tkn]; ok {
			ids[i] = id

		} else {
			ids[i] = t.TokenToIndexMap["<UNK>"] // Invalid token
		}
	}

	return ids
}

// Iterates over an array of integers (indices) and returns a string with
// corresponding tokens
func (t *Tokenizer) Decode(ids []int) string {
	tokens := make([]string, len(ids))
	for i, id := range ids {
		if tkn, ok := t.IndexToTokenMap[id]; ok {
			tokens[i] = tkn

		} else {
			tokens[i] = "<INV>"
		}
	}

	return strings.Join(tokens, " ")
}
