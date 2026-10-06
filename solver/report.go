package solver

import "math"

// Report is the archivable verdict record of a solve. It contains no large
// vectors (the solution stays in Result.X); every field is safe to embed in
// a batch summary or encode as JSON.
//
// JSON tags are provided so a batch summary can be serialized directly.
type Report struct {
	// Name identifies the task inside a batch (empty for a bare solve).
	Name string `json:"name"`
	// Method is the algorithm actually used ("cg" or "gmres").
	Method string `json:"method"`
	// Status is the machine-readable outcome ("converged", ...).
	Status string `json:"status"`
	// Success is true iff Status == converged AND the measured final
	// residual was within tolerance. It must never be true otherwise.
	Success bool `json:"success"`
	// Iterations is the total number of matrix-vector steps actually taken.
	Iterations int `json:"iterations"`
	// SegmentIterations is the number of steps advanced by the last call
	// (0 for a batch result reused from a previous run).
	SegmentIterations int `json:"segment_iterations"`
	// Tolerance is the relative residual threshold that was applied.
	Tolerance float64 `json:"tolerance"`
	// FinalRelRes is the final MEASURED relative residual ||r||/||b||.
	FinalRelRes float64 `json:"final_rel_res"`
	// ResidualMeasured is false when no trustworthy measured residual
	// exists (invalid input, or non-finite values from a breakdown).
	ResidualMeasured bool `json:"residual_measured"`
	// ResidualMet is true iff a finite measured FinalRelRes <= Tolerance.
	ResidualMet bool `json:"residual_met"`
	// History is the complete relative residual trajectory: index 0 is the
	// initial residual, then one entry per iteration, plus explicit
	// restart/segment-boundary measurements for GMRES.
	History []float64 `json:"history"`
	// Basis is the explicit decision basis for Status.
	Basis string `json:"basis"`
	// Detail carries the error detail for invalid input / breakdown /
	// cancellation / skip; empty when converged.
	Detail string `json:"detail,omitempty"`
}

// methodName renders the algorithm for a record.
func methodName(m Method) string {
	switch m {
	case MethodCG:
		return "cg"
	case MethodGMRES:
		return "gmres"
	default:
		return "auto"
	}
}

// Report produces the archivable verdict record for this result. It never
// mutates the Result. The record is complete for EVERY outcome (including
// invalid input and cancellation): iteration count, measured residual,
// full history and the decision basis are all populated whenever meaningful.
func (r *Result) Report() Report { return r.reportFor("") }

// reportFor stamps a task name into the record (used by the batch runner).
func (r *Result) reportFor(name string) Report {
	rep := Report{
		Name:              name,
		Method:            methodName(r.recordMethod),
		Status:            r.Status.String(),
		Iterations:        r.Iterations,
		SegmentIterations: r.SegmentIterations,
		Tolerance:         r.recordTol,
		FinalRelRes:       r.FinalRelRes,
		History:           append([]float64(nil), r.History...),
		Basis:             r.Basis,
		Detail:            r.ErrorMessage,
	}
	measured := !math.IsNaN(r.FinalRelRes) && !math.IsInf(r.FinalRelRes, 0)
	rep.ResidualMeasured = measured && r.Status != StatusInvalidInput
	if rep.ResidualMeasured {
		rep.ResidualMet = r.FinalRelRes <= r.toleranceOfRecord()
	}
	rep.Success = r.Status == StatusConverged && rep.ResidualMeasured && rep.ResidualMet
	return rep
}

// toleranceOfRecord recovers the applied tolerance from the basis is not
// reliable; Result carries it separately via the embed below.
func (r *Result) toleranceOfRecord() float64 { return r.recordTol }
