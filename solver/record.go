package solver

import (
	"math"
	"strconv"
)

// Record is the archivable verdict of a single solve: how many steps were
// actually taken, the final measured residual, the convergence trend and the
// basis for the verdict. It is JSON-serializable so it can be embedded
// directly into batch summaries.
//
// Converged is true only when the solve reported StatusConverged AND the
// measured final relative residual met the tolerance; a result that did not
// genuinely satisfy the residual requirement can never look successful.
type Record struct {
	// Status is the machine-readable verdict ("converged", "max_iterations",
	// "breakdown", "invalid_input", "canceled", "paused", "skipped").
	Status string `json:"status"`
	// Converged reports whether the residual requirement was truly met.
	Converged bool `json:"converged"`
	// Iterations is the number of matrix-vector steps actually taken.
	Iterations int `json:"iterations"`
	// FinalRelRes is the final measured relative residual ||r||/||b||.
	FinalRelRes float64 `json:"final_rel_res"`
	// Tolerance is the threshold the verdict was judged against.
	Tolerance float64 `json:"tolerance"`
	// History is the convergence trend: relative residual at initialization
	// and after each step that made progress.
	History []float64 `json:"history"`
	// Reason is the human-readable basis for the verdict.
	Reason string `json:"reason"`
}

// Record builds the archivable verdict for this result. The verdict is
// derived from the measured residual, not from the status flag alone: a
// Result whose measured residual does not meet the tolerance yields
// Converged == false even if its Status were inconsistent.
func (r *Result) Record() Record {
	if r == nil {
		return Record{Status: "unknown", Reason: "no result was produced"}
	}
	rec := Record{
		Status:      r.Status.String(),
		Iterations:  r.Iterations,
		FinalRelRes: r.FinalRelRes,
		Tolerance:   r.Tolerance,
		History:     append([]float64(nil), r.History...),
	}
	met := !math.IsNaN(r.FinalRelRes) && r.FinalRelRes <= r.Tolerance && r.Tolerance > 0
	rec.Converged = r.Status == StatusConverged && met
	rec.Reason = r.verdictReason(rec.Converged)
	return rec
}

// verdictReason explains why the solve ended the way it did.
func (r *Result) verdictReason(converged bool) string {
	res := strconv.FormatFloat(r.FinalRelRes, 'g', -1, 64)
	tol := strconv.FormatFloat(r.Tolerance, 'g', -1, 64)
	detail := func() string {
		if r.ErrorMessage != "" {
			return ": " + r.ErrorMessage
		}
		return ""
	}
	switch r.Status {
	case StatusConverged:
		if converged {
			return "measured relative residual " + res + " <= tolerance " + tol +
				" after " + strconv.Itoa(r.Iterations) + " iterations"
		}
		// Defensive: an inconsistent converged flag is never reported as success.
		return "convergence flag rejected: measured relative residual " + res +
			" does not satisfy tolerance " + tol
	case StatusMaxIterations:
		return "iteration budget exhausted after " + strconv.Itoa(r.Iterations) +
			" iterations with relative residual " + res + " > tolerance " + tol + detail()
	case StatusBreakdown:
		return "algorithmic breakdown after " + strconv.Itoa(r.Iterations) +
			" iterations with relative residual " + res + detail()
	case StatusInvalidInput:
		return "input rejected before solving" + detail()
	case StatusCanceled:
		return "canceled after " + strconv.Itoa(r.Iterations) +
			" iterations with relative residual " + res + detail()
	case StatusPaused:
		return "segment budget exhausted after " + strconv.Itoa(r.Iterations) +
			" total iterations with relative residual " + res +
			"; resumable from the returned checkpoint"
	case StatusSkipped:
		return "batch budget exhausted before this task started" + detail()
	default:
		return "unrecognized status" + detail()
	}
}
