package solver

import (
	"context"
	"errors"
	"math"

	"github.com/lisama569/sparse-iterative-solver-go/sparse"
)

// ErrProgressMismatch is returned when a Progress is continued with inputs
// that do not match the ones that produced it (different dimension, method,
// tolerance, ...). The handed-in Progress is never mutated, so an erroneous
// handoff cannot corrupt the progress already earned.
var ErrProgressMismatch = errors.New("progress does not match the supplied inputs or configuration")

// Progress is an opaque, resumable checkpoint of a partially completed solve.
// It is produced by Advance (or Solve when canceled) and consumed by a later
// Advance; callers must not construct or inspect one field by field.
//
// A checkpoint always corresponds to an explicitly measured iterate: the X
// returned with it and the residual reported by Residual() are consistent.
// Copying a *Progress is safe. The caller owns it exclusively and must not
// feed the same Progress to concurrent Advance calls.
type Progress struct {
	state progressState
}

// Iterations returns the total number of matrix-vector steps already taken.
func (p *Progress) Iterations() int {
	if p == nil || p.state == nil {
		return 0
	}
	return p.state.iterations()
}

// Residual returns the measured relative residual ||r||/||b|| (absolute when
// ||b|| == 0) at the checkpoint.
func (p *Progress) Residual() float64 {
	if p == nil || p.state == nil {
		return 0
	}
	return p.state.relRes()
}

// Method reports which algorithm the checkpoint belongs to.
func (p *Progress) Method() Method {
	if p == nil || p.state == nil {
		return MethodAuto
	}
	return p.state.method()
}

// Advance drives a solve forward by at most segmentSteps matrix-vector
// steps.
//
// The first segment starts with prog == nil (using cfg.InitialGuess); later
// segments pass the Progress returned by a preceding segment. segmentSteps
// is the cap for THIS call; cfg.MaxIterations is the TOTAL cap. When the
// segment cap is hit before a terminal conclusion, Advance returns
// StatusPaused with a non-nil Progress. Cancellation likewise returns
// everything computed so far (StatusCanceled with non-nil Progress), so a
// canceled segment can be resumed unchanged. Terminal statuses return nil
// progress.
//
// A segmented solve is arithmetically identical to a single one-shot solve
// with the same total budget: the same status, total iteration count and
// final residual up to floating-point rounding at measurement boundaries.
//
// Invalid input (including a mismatched Progress) returns StatusInvalidInput
// and a non-nil classified error; no progress is altered.
func Advance(ctx context.Context, A *sparse.CSR, b []float64, cfg Config, prog *Progress, segmentSteps int) (*Result, error) {
	n, err := validate(A, b, &cfg)
	if err != nil {
		return &Result{Status: StatusInvalidInput, ErrorMessage: err.Error(),
			Basis: invalidBasis(err), recordTol: cfg.Tolerance,
			recordMethod: cfg.Method}, err
	}
	if ctx == nil {
		ctx = context.Background()
	}
	if segmentSteps < 0 {
		return &Result{Status: StatusInvalidInput,
			ErrorMessage: "segment steps must be non-negative",
			Basis:        "segment steps must be non-negative",
			recordTol:    cfg.Tolerance, recordMethod: cfg.Method}, nil
	}

	// Resume or start.
	var ps *cgState
	var gs *gmresState
	startedIter := 0
	if prog != nil {
		switch st := prog.state.(type) {
		case *cgState:
			if err := st.checkCompatible(cfg, n); err != nil {
				return &Result{Status: StatusInvalidInput,
					ErrorMessage: err.Error(), Basis: err.Error()}, err
			}
			// Work on a private copy: a failed/canceled continuation must
			// never mutate the handed-in checkpoint.
			ps = st.clone()
			startedIter = ps.iter
		case *gmresState:
			if err := st.checkCompatible(cfg, n); err != nil {
				return &Result{Status: StatusInvalidInput,
					ErrorMessage: err.Error(), Basis: err.Error()}, err
			}
			gs = st.clone()
			startedIter = gs.iter
		default:
			err := ErrProgressMismatch
			return &Result{Status: StatusInvalidInput,
				ErrorMessage: err.Error(), Basis: err.Error()}, err
		}
		if startedIter >= cfg.MaxIterations {
			// Nothing left to do: the checkpoint already used the full
			// budget. Report the measured state honestly.
			res := finalizeState(prog.state, StatusMaxIterations, cfg)
			res.Basis = maxItersBasis(cfg.MaxIterations, res.FinalRelRes, cfg.Tolerance)
			return res, nil
		}
	} else {
		x0 := make([]float64, n)
		if cfg.InitialGuess != nil {
			copy(x0, cfg.InitialGuess)
		}
		switch cfg.Method {
		case MethodCG:
			st, initStatus, ierr := newCGState(cfg, A, b, x0)
			if ierr != nil {
				return &Result{Status: StatusInvalidInput, X: x0,
					ErrorMessage: ierr.Error(), Basis: invalidBasis(ierr)}, ierr
			}
			if initStatus == StatusConverged {
				res := finalizeState(st, StatusConverged, cfg)
				res.Basis = convergedBasis(res.FinalRelRes, cfg.Tolerance, res.Iterations)
				return res, nil
			}
			if initStatus == StatusBreakdown {
				res := finalizeState(st, StatusBreakdown, cfg)
				res.Basis = breakdownBasis(res.ErrorMessage)
				return res, nil
			}
			ps = st
		case MethodGMRES:
			st, initStatus, ierr := newGMRESState(cfg, A, b, x0)
			if ierr != nil {
				return &Result{Status: StatusInvalidInput, X: x0,
					ErrorMessage: ierr.Error(), Basis: invalidBasis(ierr)}, ierr
			}
			if initStatus == StatusConverged {
				res := finalizeState(st, StatusConverged, cfg)
				res.Basis = convergedBasis(res.FinalRelRes, cfg.Tolerance, res.Iterations)
				return res, nil
			}
			gs = st
		default:
			return &Result{Status: StatusInvalidInput,
				ErrorMessage: "unknown method", Basis: "unknown method"}, nil
		}
	}

	// How many steps may THIS segment take?
	remaining := cfg.MaxIterations - startedIter
	steps := remaining
	if segmentSteps > 0 && segmentSteps < steps {
		steps = segmentSteps
	}

	var status Status
	var st progressState
	if ps != nil {
		st = ps
		status = ps.advance(ctx, A, b, steps, steps == remaining)
	} else {
		st = gs
		status = gs.advance(ctx, A, b, steps, steps == remaining)
	}

	if status == StatusInvalidInput {
		err := errors.New(st.detail())
		return &Result{Status: StatusInvalidInput,
			ErrorMessage: err.Error(), Basis: invalidBasis(err)}, err
	}

	res := finalizeState(st, status, cfg)
	res.SegmentIterations = res.Iterations - startedIter

	switch status {
	case StatusConverged:
		// For GMRES this may be a projected claim: finalizeState remeasured
		// the true residual, and the driver below enforces it.
		res.Basis = convergedBasis(res.FinalRelRes, cfg.Tolerance, res.Iterations)
	case StatusMaxIterations:
		res.Basis = maxItersBasis(cfg.MaxIterations, res.FinalRelRes, cfg.Tolerance)
	case StatusBreakdown:
		res.Basis = breakdownBasis(res.ErrorMessage)
	case StatusPaused:
		res.Basis = pausedBasis(steps, res.FinalRelRes, cfg.Tolerance, res.Iterations)
		res.Progress = checkpoint(st)
	case StatusCanceled:
		res.Basis = canceledBasis(res.ErrorMessage, res.FinalRelRes, res.Iterations)
		res.Progress = checkpoint(st)
	}

	// Hard guarantee: nothing that did not meet the tolerance is a success.
	if res.Status == StatusConverged && !(res.FinalRelRes <= cfg.Tolerance) {
		res.Status = StatusBreakdown
		if res.Progress != nil {
			res.Progress = nil
		}
		res.ErrorMessage = "internal: convergence claim inconsistent with measured residual"
		res.Basis = breakdownBasis(res.ErrorMessage)
	}
	if math.IsNaN(res.FinalRelRes) && res.Status == StatusConverged {
		res.Status = StatusBreakdown
	}
	return res, nil
}

// finalizeState copies the checkpoint contents into a Result. The returned
// X is a private copy so callers can never mutate solver state, and
// converged GMRES claims are backed by an explicit residual measurement.
func finalizeState(st progressState, status Status, cfg Config) *Result {
	switch s := st.(type) {
	case *cgState:
		res := &Result{
			Status:       status,
			X:            cloneFloats(s.x),
			Iterations:   s.iter,
			FinalRelRes:  s.rel,
			History:      append([]float64(nil), s.history...),
			ErrorMessage: s.detail(),
			recordTol:    cfg.Tolerance, recordMethod: cfg.Method,
		}
		return res
	case *gmresState:
		res := &Result{
			Status:       status,
			X:            cloneFloats(s.x),
			Iterations:   s.iter,
			FinalRelRes:  s.rel,
			History:      append([]float64(nil), s.history...),
			ErrorMessage: s.detail(),
			recordTol:    cfg.Tolerance, recordMethod: cfg.Method,
		}
		return res
	default:
		return &Result{Status: status}
	}
}

func cloneFloats(v []float64) []float64 {
	out := make([]float64, len(v))
	copy(out, v)
	return out
}

// progressState is the method-specific checkpoint interface.
type progressState interface {
	iterations() int
	relRes() float64
	method() Method
	detail() string
}

func (s *cgState) detail() string { return s.errDetail }

// checkpoint wraps a deep copy of a method state so the returned Progress
// is a frozen snapshot independent of the working state.
func checkpoint(st progressState) *Progress {
	switch s := st.(type) {
	case *cgState:
		return &Progress{state: s.clone()}
	case *gmresState:
		return &Progress{state: s.clone()}
	default:
		return &Progress{state: st}
	}
}
