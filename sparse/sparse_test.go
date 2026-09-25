package sparse

import (
	"errors"
	"math"
	"testing"
)

func TestCOOToCSRGeneral(t *testing.T) {
	c := &COO{Rows: 2, Cols: 2, Triplets: []Triplet{
		{Row: 0, Col: 0, Value: 4}, {Row: 0, Col: 1, Value: 1}, {Row: 1, Col: 0, Value: 1}, {Row: 1, Col: 1, Value: 3},
	}}
	m, err := c.ToCSR()
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if m.NNZ() != 4 {
		t.Fatalf("nnz = %d, want 4", m.NNZ())
	}
	y, err := m.MatVec([]float64{1, 2})
	if err != nil {
		t.Fatal(err)
	}
	if y[0] != 6 || y[1] != 7 {
		t.Fatalf("matvec = %v, want [6 7]", y)
	}
}

func TestSymmetricExpansion(t *testing.T) {
	c := &COO{Rows: 2, Cols: 2, Symmetric: true, Triplets: []Triplet{
		{Row: 0, Col: 0, Value: 2}, {Row: 1, Col: 0, Value: 5}, {Row: 1, Col: 1, Value: 1}, // lower triangle incl. diagonal
	}}
	m, err := c.ToCSR()
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if m.NNZ() != 4 {
		t.Fatalf("nnz = %d, want 4", m.NNZ())
	}
	y, _ := m.MatVec([]float64{1, 1})
	if y[0] != 7 || y[1] != 6 {
		t.Fatalf("matvec = %v, want [7 6]", y)
	}
}

func TestErrorsClassified(t *testing.T) {
	cases := []struct {
		name string
		coo  *COO
		kind ErrorKind
	}{
		{"bad dims", &COO{Rows: 0, Cols: 2}, KindStructure},
		{"nonsquare symmetric", &COO{Rows: 2, Cols: 3, Symmetric: true}, KindStructure},
		{"bad index", &COO{Rows: 2, Cols: 2, Triplets: []Triplet{{Row: 2, Col: 0, Value: 1}}}, KindStructure},
		{"duplicate", &COO{Rows: 2, Cols: 2, Triplets: []Triplet{{Row: 0, Col: 0, Value: 1}, {Row: 0, Col: 0, Value: 2}}}, KindStructure},
		{"symmetric pair duplicate", &COO{Rows: 2, Cols: 2, Symmetric: true,
			Triplets: []Triplet{{Row: 0, Col: 1, Value: 1}, {Row: 1, Col: 0, Value: 2}}}, KindStructure},
		{"nan", &COO{Rows: 2, Cols: 2, Triplets: []Triplet{{Row: 0, Col: 0, Value: math.NaN()}}}, KindNumeric},
		{"inf", &COO{Rows: 2, Cols: 2, Triplets: []Triplet{{0, 0, math.Inf(1)}}}, KindNumeric},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			_, err := tc.coo.ToCSR()
			var ie *InputError
			if !errors.As(err, &ie) {
				t.Fatalf("expected *InputError, got %v", err)
			}
			if ie.Kind != tc.kind {
				t.Fatalf("kind = %d, want %d (%v)", ie.Kind, tc.kind, ie)
			}
		})
	}
}

func TestMatVecShapeMismatch(t *testing.T) {
	m, _ := (&COO{Rows: 2, Cols: 2, Triplets: []Triplet{{Row: 0, Col: 0, Value: 1}, {Row: 1, Col: 1, Value: 1}}}).ToCSR()
	if _, err := m.MatVec([]float64{1}); err == nil {
		t.Fatal("expected shape mismatch error")
	}
}
