package solver

import (
	"errors"
	"fmt"
	"math"

	"github.com/lisama569/sparse-iterative-solver-go/sparse"
)

// Basis strings are the explicit decision basis carried in Result.Basis /
// Report.Basis: reading them must make clear exactly why the status was
// assigned, including the measured number and the threshold.

func convergedBasis(rel, tol float64, iters int) string {
	return fmt.Sprintf("measured relative residual %.6e <= tolerance %.6e after %d iteration(s)", rel, tol, iters)
}

func maxItersBasis(budget int, rel, tol float64) string {
	relStr := floatStr(rel)
	if math.IsNaN(rel) {
		relStr = "NaN"
	}
	return fmt.Sprintf("iteration budget %d exhausted with measured relative residual %s > tolerance %.6e",
		budget, relStr, tol)
}

func breakdownBasis(detail string) string {
	if detail == "" {
		return "algorithmic breakdown; the residual requirement was not met"
	}
	return "algorithmic breakdown: " + detail
}

func pausedBasis(segmentSteps int, rel, tol float64, totalIters int) string {
	return fmt.Sprintf("segment step cap %d reached before convergence at %d total iteration(s); measured relative residual %.6e > tolerance %.6e; progress is resumable",
		segmentSteps, totalIters, rel, tol)
}

func canceledBasis(detail string, rel float64, iters int) string {
	if detail == "" {
		detail = "context canceled"
	}
	relStr := floatStr(rel)
	if math.IsNaN(rel) {
		relStr = "NaN"
	}
	return fmt.Sprintf("canceled after %d iteration(s) with measured relative residual %s; progress retained: %s",
		iters, relStr, detail)
}

func skippedBasis(reason string) string {
	return "task never started: " + reason
}

func invalidBasis(err error) string {
	if err == nil {
		return "invalid input"
	}
	var ie *sparse.InputError
	if errors.As(err, &ie) {
		return fmt.Sprintf("invalid input (%s): %s", ie.Kind, err.Error())
	}
	return "invalid input: " + err.Error()
}

// match reports whether the continuation-safe fields of two Configs agree.
func (c Config) match(o Config) bool {
	return c.Tolerance == o.Tolerance &&
		c.MaxIterations == o.MaxIterations &&
		c.Precondition == o.Precondition &&
		c.Restart == o.Restart
}
