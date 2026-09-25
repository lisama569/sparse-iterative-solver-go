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
)

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

// Result is the outcome of a solve. A result only reports StatusConverged
// when the final relative residual was measured to be <= Tolerance.
type Result struct {
	Status      Status
	X           []float64
	Iterations  int
	FinalRelRes float64
	// History holds relative residual norms at initialization and after
	// each iteration that made progress.
	History      []float64
	ErrorMessage string
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
	n, err := validate(A, b, &cfg)
	if err != nil {
		return &Result{Status: StatusInvalidInput, ErrorMessage: err.Error()}, err
	}
	if ctx == nil {
		ctx = context.Background()
	}
	if err := ctx.Err(); err != nil {
		return canceledResult(cfg, err), nil
	}

	x := make([]float64, n)
	if cfg.InitialGuess != nil {
		copy(x, cfg.InitialGuess)
	}

	var res *Result
	switch cfg.Method {
	case MethodCG:
		res = solveCG(ctx, A, b, x, cfg)
	case MethodGMRES:
		res = solveGMRES(ctx, A, b, x, cfg)
	default:
		res = &Result{Status: StatusInvalidInput, ErrorMessage: "unknown method"}
	}
	// Final guarantee: nothing that did not meet tolerance is "converged".
	if res.Status == StatusConverged && !(res.FinalRelRes <= cfg.Tolerance) {
		res.Status = StatusBreakdown
		if res.ErrorMessage == "" {
			res.ErrorMessage = "internal: convergence flag inconsistent with measured residual"
		}
	}
	return res, nil
}

func canceledResult(cfg Config, err error) *Result {
	return &Result{Status: StatusCanceled, ErrorMessage: err.Error()}
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
