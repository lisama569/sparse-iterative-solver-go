package solver

import (
	"math"
	"strconv"
)

func vdot(x, y []float64) float64 {
	s := 0.0
	for i := range x {
		s += x[i] * y[i]
	}
	return s
}

func vnorm(x []float64) float64 { return math.Sqrt(vdot(x, x)) }

// safeRelNorm returns ||r||/||b||; when ||b|| == 0 it returns the absolute
// residual norm (the only meaningful scale). Non-finite residuals propagate
// as NaN so they can never accidentally satisfy a tolerance comparison.
func safeRelNorm(r []float64, bNorm float64) float64 {
	n := vnorm(r)
	if math.IsNaN(n) {
		return math.NaN()
	}
	if bNorm == 0 {
		return n
	}
	return n / bNorm
}

func finiteVec(v []float64) bool {
	for _, f := range v {
		if math.IsNaN(f) || math.IsInf(f, 0) {
			return false
		}
	}
	return true
}

func floatStr(f float64) string { return strconv.FormatFloat(f, 'g', -1, 64) }

func invalidFrom(err error, x []float64) *Result {
	return &Result{Status: StatusInvalidInput, X: x, ErrorMessage: err.Error()}
}
