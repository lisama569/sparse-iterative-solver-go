// Command solver runs an iterative sparse linear solve entirely locally.
//
// Usage:
//
//	solver -matrix A.mtx [-rhs b.txt] [-method auto|cg|gmres]
//	       [-tol 1e-8] [-maxiter N] [-precond] [-x0 x0.txt]
//	       [-restart M] [-history]
//
// The matrix must be a Matrix Market coordinate file (real/integer/pattern,
// general/symmetric). The optional right-hand side is a text file with one
// floating point number per line; when omitted, b = (1,...,1) is used.
// Exit code is 0 only when the solve converges; 1 on max iterations or
// breakdown; 2 on invalid input or file errors.
package main

import (
	"bufio"
	"context"
	"errors"
	"flag"
	"fmt"
	"os"

	"github.com/lisama569/sparse-iterative-solver-go/mmio"
	"github.com/lisama569/sparse-iterative-solver-go/solver"
	"github.com/lisama569/sparse-iterative-solver-go/sparse"
)

func main() {
	matrixPath := flag.String("matrix", "", "path to a Matrix Market coordinate matrix (.mtx)")
	rhsPath := flag.String("rhs", "", "path to the right-hand side (one number per line); defaults to all ones")
	x0Path := flag.String("x0", "", "path to an initial guess (one number per line)")
	method := flag.String("method", "auto", "solver method: auto, cg, gmres")
	tol := flag.Float64("tol", 1e-8, "relative residual tolerance")
	maxIter := flag.Int("maxiter", 0, "maximum iterations (0 = 10*n)")
	restart := flag.Int("restart", 0, "GMRES restart length (0 = min(n,50))")
	precond := flag.Bool("precond", false, "enable Jacobi (diagonal) preconditioning")
	showHistory := flag.Bool("history", false, "print the full relative residual history")
	flag.Parse()

	if *matrixPath == "" {
		fmt.Fprintln(os.Stderr, "error: -matrix is required")
		flag.Usage()
		os.Exit(2)
	}

	coo, err := mmio.ReadFile(*matrixPath)
	if err != nil {
		fmt.Fprintf(os.Stderr, "error: cannot load matrix: %v\n", classify(err))
		os.Exit(2)
	}
	A, err := coo.ToCSR()
	if err != nil {
		fmt.Fprintf(os.Stderr, "error: invalid matrix: %v\n", classify(err))
		os.Exit(2)
	}
	n := A.Rows
	if A.Rows != A.Cols {
		fmt.Fprintf(os.Stderr, "error: solver requires a square matrix, got %dx%d\n", A.Rows, A.Cols)
		os.Exit(2)
	}

	b := ones(n)
	if *rhsPath != "" {
		b, err = readVector(*rhsPath, n)
		if err != nil {
			fmt.Fprintf(os.Stderr, "error: %v\n", err)
			os.Exit(2)
		}
	}
	var x0 []float64
	if *x0Path != "" {
		x0, err = readVector(*x0Path, n)
		if err != nil {
			fmt.Fprintf(os.Stderr, "error: %v\n", err)
			os.Exit(2)
		}
	}

	cfg := solver.Config{
		Tolerance:     *tol,
		MaxIterations: *maxIter,
		InitialGuess:  x0,
		Precondition:  *precond,
		Restart:       *restart,
	}
	switch *method {
	case "auto":
		cfg.Method = solver.MethodAuto
	case "cg":
		cfg.Method = solver.MethodCG
	case "gmres":
		cfg.Method = solver.MethodGMRES
	default:
		fmt.Fprintf(os.Stderr, "error: unknown method %q\n", *method)
		os.Exit(2)
	}

	res, err := solver.Solve(context.Background(), A, b, cfg)
	if err != nil {
		fmt.Fprintf(os.Stderr, "error: invalid input: %v\n", classify(err))
		os.Exit(2)
	}

	fmt.Printf("matrix: %s size=%dx%d nnz=%d symmetric=%t\n",
		*matrixPath, A.Rows, A.Cols, A.NNZ(), A.Symmetric)
	fmt.Printf("method=%s preconditioned=%t iterations=%d status=%s relative_residual=%.6e\n",
		*method, *precond, res.Iterations, res.Status, res.FinalRelRes)
	fmt.Printf("history summary: %s\n", summarize(res.History))
	if *showHistory {
		for i, h := range res.History {
			fmt.Printf("  iter %d: %.6e\n", i, h)
		}
	}

	switch res.Status {
	case solver.StatusConverged:
		os.Exit(0)
	case solver.StatusInvalidInput:
		fmt.Fprintf(os.Stderr, "error: %s\n", res.ErrorMessage)
		os.Exit(2)
	default:
		fmt.Fprintf(os.Stderr, "solve did not converge: %s (%s)\n", res.Status, res.ErrorMessage)
		os.Exit(1)
	}
}

func ones(n int) []float64 {
	v := make([]float64, n)
	for i := range v {
		v[i] = 1
	}
	return v
}

func readVector(path string, n int) ([]float64, error) {
	f, err := os.Open(path)
	if err != nil {
		return nil, fmt.Errorf("cannot open %q: %w", path, err)
	}
	defer f.Close()
	var v []float64
	sc := bufio.NewScanner(f)
	sc.Buffer(make([]byte, 0, 64*1024), 16*1024*1024)
	lineNo := 0
	for sc.Scan() {
		lineNo++
		text := sc.Text()
		// Strip and ignore comments and blanks.
		if i := indexByte(text, '%'); i >= 0 {
			text = text[:i]
		}
		if trimSpace(text) == "" {
			continue
		}
		var x float64
		if _, err := parseFloat(trimSpace(text), &x); err != nil {
			return nil, fmt.Errorf("%s line %d: %w", path, lineNo, err)
		}
		v = append(v, x)
	}
	if err := sc.Err(); err != nil {
		return nil, fmt.Errorf("reading %q: %w", path, err)
	}
	if len(v) != n {
		return nil, &sparse.InputError{Kind: sparse.KindStructure,
			Message: fmt.Sprintf("%s has %d entries, expected %d", path, len(v), n)}
	}
	return v, nil
}

func classify(err error) error {
	var ie *sparse.InputError
	if errors.As(err, &ie) {
		return fmt.Errorf("[%s] %s", kindName(ie.Kind), ie.Message)
	}
	return err
}

func kindName(k sparse.ErrorKind) string {
	switch k {
	case sparse.KindFormat:
		return "format"
	case sparse.KindStructure:
		return "structure"
	case sparse.KindNumeric:
		return "numeric"
	default:
		return "error"
	}
}

func summarize(h []float64) string {
	if len(h) == 0 {
		return "<empty>"
	}
	return fmt.Sprintf("len=%d first=%.3e last=%.3e", len(h), h[0], h[len(h)-1])
}
