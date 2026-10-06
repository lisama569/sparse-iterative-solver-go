package solver

import (
	"context"

	"github.com/lisama569/sparse-iterative-solver-go/sparse"
)

// Checkpoint captures resumable solver progress between segments. It is
// JSON-serializable so it can be persisted and handed off between processes.
//
// A checkpoint is only meaningful for the problem and configuration it was
// created from; SolveSegment rejects resuming with a mismatched matrix
// dimension, method, tolerance, restart length or preconditioner setting.
type Checkpoint struct {
	// Method is the algorithm that produced this checkpoint.
	Method Method `json:"method"`
	// N is the problem dimension.
	N int `json:"n"`
	// Iterations is the total number of iterations consumed so far, across
	// all completed segments.
	Iterations int `json:"iterations"`
	// Tolerance is the relative residual threshold being solved for.
	Tolerance float64 `json:"tolerance"`
	// Restart is the GMRES restart length (0 for CG).
	Restart int `json:"restart,omitempty"`
	// Precondition records whether Jacobi preconditioning is active.
	Precondition bool `json:"precondition,omitempty"`
	// Rel is the last measured relative residual.
	Rel float64 `json:"rel"`
	// History is the cumulative convergence trend so far.
	History []float64 `json:"history"`
	// X is the current approximate solution.
	X []float64 `json:"x"`
	// R, P and Rz carry the CG recurrence state (nil/zero for GMRES, which
	// checkpoints at restart boundaries and only needs X).
	R  []float64 `json:"r,omitempty"`
	P  []float64 `json:"p,omitempty"`
	Rz float64   `json:"rz,omitempty"`
}

// SolveSegment advances a solve of A*x = b by at most segmentBudget
// iterations and returns a checkpoint from which the solve can be resumed.
//
//   - prev == nil starts a fresh solve (cfg.InitialGuess is honored);
//   - prev != nil resumes from a checkpoint produced by an earlier
//     SolveSegment call for the same problem and configuration.
//
// The returned Result reports cumulative iterations and history (not just
// this segment's). When the status is StatusPaused (segment budget spent) or
// StatusCanceled (ctx canceled mid-segment), the returned Checkpoint is
// non-nil and preserves all progress made so far; terminal statuses
// (converged, max iterations, breakdown, invalid input) return a nil
// checkpoint. A resumed solve follows the same trajectory as an
// uninterrupted one, so segmented and one-shot conclusions agree.
func SolveSegment(ctx context.Context, A *sparse.CSR, b []float64, cfg Config, prev *Checkpoint, segmentBudget int) (*Result, *Checkpoint, error) {
	if segmentBudget < 0 {
		err := &sparse.InputError{Kind: sparse.KindStructure, Message: "segment budget must be non-negative"}
		return &Result{Status: StatusInvalidInput, ErrorMessage: err.Error()}, prev, err
	}
	n, err := validate(A, b, &cfg)
	if err != nil {
		return &Result{Status: StatusInvalidInput, Tolerance: cfg.Tolerance, ErrorMessage: err.Error()}, prev, err
	}
	if ctx == nil {
		ctx = context.Background()
	}
	if prev != nil {
		if err := checkResume(cfg, prev, n); err != nil {
			return &Result{Status: StatusInvalidInput, Tolerance: cfg.Tolerance, ErrorMessage: err.Error()}, prev, err
		}
	}
	if err := ctx.Err(); err != nil {
		res := canceledResult(cfg, err)
		if prev != nil {
			res.Iterations = prev.Iterations
			res.FinalRelRes = prev.Rel
			res.History = append([]float64(nil), prev.History...)
		}
		return res, prev, nil
	}

	var res *Result
	var ck *Checkpoint
	switch cfg.Method {
	case MethodCG:
		c := prev
		if c == nil {
			x := make([]float64, n)
			copy(x, cfg.InitialGuess)
			var term *Result
			c, term = newCGCheckpoint(A, b, x, cfg)
			if term != nil {
				term.Tolerance = cfg.Tolerance
				return term, nil, nil
			}
		} else {
			c = cloneCheckpoint(c)
		}
		res, ck = cgRun(ctx, A, b, cfg, c, segmentBudget)
	case MethodGMRES:
		c := prev
		if c == nil {
			x := make([]float64, n)
			copy(x, cfg.InitialGuess)
			var term *Result
			c, term = newGMRESCheckpoint(A, b, x, cfg)
			if term != nil {
				term.Tolerance = cfg.Tolerance
				return term, nil, nil
			}
		} else {
			c = cloneCheckpoint(c)
		}
		res, ck = gmresRun(ctx, A, b, cfg, c, segmentBudget)
	default:
		res = &Result{Status: StatusInvalidInput, ErrorMessage: "unknown method"}
	}
	res.Tolerance = cfg.Tolerance
	// Final guarantee: nothing that did not meet tolerance is "converged".
	if res.Status == StatusConverged && !(res.FinalRelRes <= cfg.Tolerance) {
		res.Status = StatusBreakdown
		if res.ErrorMessage == "" {
			res.ErrorMessage = "internal: convergence flag inconsistent with measured residual"
		}
		ck = nil
	}
	return res, ck, nil
}

// checkResume verifies that a checkpoint matches the problem and normalized
// configuration it is about to resume with.
func checkResume(cfg Config, prev *Checkpoint, n int) error {
	bad := func(msg string) error {
		return &sparse.InputError{Kind: sparse.KindStructure, Message: "checkpoint mismatch: " + msg}
	}
	if prev.N != n {
		return bad("checkpoint dimension does not match the matrix")
	}
	if prev.Method != cfg.Method {
		return bad("checkpoint method " + prev.Method.String() + " != requested " + cfg.Method.String())
	}
	if prev.Tolerance != cfg.Tolerance {
		return bad("checkpoint tolerance differs from requested tolerance")
	}
	if prev.Precondition != cfg.Precondition {
		return bad("checkpoint preconditioner setting differs")
	}
	if cfg.Method == MethodGMRES && prev.Restart != cfg.Restart {
		return bad("checkpoint GMRES restart length differs")
	}
	if len(prev.X) != n {
		return bad("checkpoint solution vector has wrong length")
	}
	if prev.Method == MethodCG && prev.Iterations > 0 && (len(prev.R) != n || len(prev.P) != n) {
		return bad("CG checkpoint recurrence state has wrong length")
	}
	return nil
}

// cloneCheckpoint deep-copies c so resuming never mutates the caller's
// checkpoint (a failed or canceled handoff cannot corrupt saved progress).
func cloneCheckpoint(c *Checkpoint) *Checkpoint {
	out := *c
	out.History = append([]float64(nil), c.History...)
	out.X = append([]float64(nil), c.X...)
	out.R = append([]float64(nil), c.R...)
	out.P = append([]float64(nil), c.P...)
	return &out
}

// String renders a method for logs and error messages.
func (m Method) String() string {
	switch m {
	case MethodCG:
		return "cg"
	case MethodGMRES:
		return "gmres"
	case MethodAuto:
		return "auto"
	default:
		return "unknown"
	}
}
