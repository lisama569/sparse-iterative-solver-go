package solver

import (
	"context"
	"math"
	"strconv"

	"github.com/lisama569/sparse-iterative-solver-go/sparse"
)

// Status classifies the outcome of an iterative solve.
type Status int

const (
	// StatusConverged means the residual requirement was met.
	StatusConverged Status = iota + 1
	// StatusMaxIterations means the step budget was exhausted while the
	// residual was still above the tolerance.
	StatusMaxIterations
	// StatusBreakdown means the algorithm detected a fatal numerical
	// breakdown (zero/non-positive direction for CG, zero Krylov basis for
	// GMRES) or produced non-finite values.
	StatusBreakdown
	// StatusInvalidInput means the request could not start because the
	// matrix, right-hand side or configuration were invalid.
	StatusInvalidInput
	// StatusCanceled means the context was canceled mid-solve.
	StatusCanceled
	// StatusPaused means a segment reached its per-segment step cap before a
	// terminal conclusion; the returned Progress lets a later Advance call
	// continue. A paused solve is never a success: the residual was still
	// above the tolerance.
	StatusPaused
	// StatusSkipped is only produced by batch runners (never by Solve): the
	// task never started because the batch budget or deadline was exhausted,
	// or the batch context was canceled before its turn.
	StatusSkipped
)

// Terminal reports whether the status is a final conclusion. Only
// StatusConverged is a success; StatusPaused is resumable, StatusSkipped
// means "never started".
func (s Status) Terminal() bool {
	switch s {
	case StatusConverged, StatusMaxIterations, StatusBreakdown, StatusInvalidInput, StatusCanceled:
		return true
	default:
		return false
	}
}

// String renders a status for logs and error messages.
func (s Status) String() string {
	switch s {
	case StatusConverged:
		return "converged"
	case StatusMaxIterations:
		return "max_iterations"
	case StatusBreakdown:
		return "breakdown"
	case StatusInvalidInput:
		return "invalid_input"
	case StatusCanceled:
		return "canceled"
	case StatusPaused:
		return "paused"
	case StatusSkipped:
		return "skipped"
	default:
		return "unknown"
	}
}

// Method selects an iterative algorithm.
type Method int

const (
	// MethodAuto picks CG for symmetric systems and GMRES otherwise.
	MethodAuto Method = iota
	MethodCG
	MethodGMRES
)

// Config configures an iterative solve.
type Config struct {
	// Tolerance is the relative residual threshold ||r||/||b||.
	Tolerance float64
	// MaxIterations is the maximum number of matrix-vector steps;
	// 0 selects a default of 10*n.
	MaxIterations int
	// InitialGuess is the starting vector; nil means the zero vector.
	InitialGuess []float64
	// Precondition enables the Jacobi (diagonal) preconditioner.
	Precondition bool
	// Restart is the GMRES restart length; 0 selects min(n, 50).
	Restart int
	// Method selects the algorithm; MethodAuto uses symmetry metadata.
	Method Method
}

// Result is the outcome of a solve (or of one segment). A result only
// reports StatusConverged when the final relative residual was measured to
// be <= Tolerance.
type Result struct {
	Status      Status
	X           []float64
	Iterations  int
	FinalRelRes float64
	// SegmentIterations is the number of steps advanced by this call alone
	// (0 for batch results that were reused from a previous batch run).
	SegmentIterations int
	// History holds relative residual norms at initialization and after
	// each iteration that made progress. For segmented solves every segment
	// shares the cumulative history, so the complete trajectory is available
	// even when it is assembled from several calls.
	History      []float64
	ErrorMessage string
	// Basis is a short machine/human-readable statement of why Status was
	// reached ("residual 1.2e-9 <= tolerance 1e-8", "iteration budget 100
	// exhausted with residual ...", ...).
	Basis string
	// Progress, when non-nil, carries resumable solver state. It is set for
	// StatusPaused and StatusCanceled (a canceled segment keeps everything
	// computed so far) and must be nil for every terminal status.
	Progress *Progress

	// recordTol is the tolerance actually applied; it backs Report so an
	// archived record can judge the residual without the original Config.
	recordTol float64
	// recordMethod is the algorithm actually used.
	recordMethod Method
}

// DefaultConfig returns a usable default configuration (tol 1e-8,
// auto method, no preconditioning).
func DefaultConfig() Config {
	return Config{Tolerance: 1e-8}
}

// Solve solves A*x = b.
//
// Invalid input yields a Result with Status StatusInvalidInput and a
// non-nil error carrying a sparse.ErrorKind; algorithmic outcomes
// (converged / max iterations / breakdown / canceled) return a nil error
// with the corresponding status. The matrix is only read, so the same
// *sparse.CSR can safely be shared between concurrent Solve calls.
func Solve(ctx context.Context, A *sparse.CSR, b []float64, cfg Config) (*Result, error) {
	if ctx == nil {
		ctx = context.Background()
	}
	// One-shot solve is exactly one segment with no per-segment cap; the
	// total cfg.MaxIterations budget applies.
	res, err := Advance(ctx, A, b, cfg, nil, 0)
	if res == nil {
		return res, err
	}
	// A one-shot solve can never legitimately pause: no segment cap.
	if res.Status == StatusPaused {
		res.Status = StatusMaxIterations
		res.Basis = maxItersBasis(cfg.MaxIterations, res.FinalRelRes, res.recordTol)
		res.Progress = nil
	}
	return res, err
}

// validate normalizes cfg and checks the inputs. It returns the dimension n.
func validate(A *sparse.CSR, b []float64, cfg *Config) (int, error) {
	if A == nil {
		return 0, &sparse.InputError{Kind: sparse.KindStructure, Message: "nil matrix"}
	}
	n := A.Rows
	if n != A.Cols {
		return 0, &sparse.InputError{Kind: sparse.KindStructure,
			Message: "solver requires a square matrix, got " + strconv.Itoa(n) + "x" + strconv.Itoa(A.Cols)}
	}
	if len(b) != n {
		return 0, &sparse.InputError{Kind: sparse.KindStructure,
			Message: "right-hand side length " + strconv.Itoa(len(b)) + " does not match dimension " + strconv.Itoa(n)}
	}
	for _, v := range b {
		if math.IsNaN(v) || math.IsInf(v, 0) {
			return 0, &sparse.InputError{Kind: sparse.KindNumeric, Message: "right-hand side contains non-finite values"}
		}
	}
	if cfg.InitialGuess != nil && len(cfg.InitialGuess) != n {
		return 0, &sparse.InputError{Kind: sparse.KindStructure,
			Message: "initial guess length " + strconv.Itoa(len(cfg.InitialGuess)) + " does not match dimension " + strconv.Itoa(n)}
	}
	for _, v := range cfg.InitialGuess {
		if math.IsNaN(v) || math.IsInf(v, 0) {
			return 0, &sparse.InputError{Kind: sparse.KindNumeric, Message: "initial guess contains non-finite values"}
		}
	}
	if cfg.Tolerance == 0 {
		cfg.Tolerance = 1e-8
	}
	if math.IsNaN(cfg.Tolerance) || math.IsInf(cfg.Tolerance, 0) || cfg.Tolerance <= 0 {
		return 0, &sparse.InputError{Kind: sparse.KindStructure, Message: "tolerance must be a finite positive number"}
	}
	if cfg.MaxIterations == 0 {
		cfg.MaxIterations = 10 * n
	}
	if cfg.MaxIterations < 0 {
		return 0, &sparse.InputError{Kind: sparse.KindStructure, Message: "max iterations must be non-negative"}
	}
	if cfg.Restart == 0 {
		cfg.Restart = min(n, 50)
	}
	if cfg.Restart < 0 {
		return 0, &sparse.InputError{Kind: sparse.KindStructure, Message: "restart must be non-negative"}
	}
	switch cfg.Method {
	case MethodAuto:
		if A.Symmetric {
			cfg.Method = MethodCG
		} else {
			cfg.Method = MethodGMRES
		}
	case MethodCG:
		// CG's conjugacy derivation assumes a symmetric (SPD) operator.
		if !A.Symmetric {
			return 0, &sparse.InputError{Kind: sparse.KindStructure,
				Message: "CG requires a symmetric matrix; use MethodGMRES or MethodAuto for a general matrix"}
		}
	case MethodGMRES:
	default:
		return 0, &sparse.InputError{Kind: sparse.KindStructure, Message: "unknown solver method"}
	}
	// Jacobi preconditioner needs nonzero, finite diagonal entries.
	if cfg.Precondition {
		diag := A.Diagonal()
		for i, d := range diag {
			if d == 0 {
				return 0, &sparse.InputError{Kind: sparse.KindNumeric,
					Message: "Jacobi preconditioner requires nonzero diagonal (zero at row " + strconv.Itoa(i) + ")"}
			}
			if math.IsNaN(d) || math.IsInf(d, 0) {
				return 0, &sparse.InputError{Kind: sparse.KindNumeric,
					Message: "diagonal contains non-finite entry at row " + strconv.Itoa(i)}
			}
		}
	}
	return n, nil
}
