package solver

import (
	"context"
	"encoding/json"
	"errors"
	"math"
	"testing"
	"time"

	"github.com/lisama569/sparse-iterative-solver-go/sparse"
)

// TestReportCompleteForEveryOutcome verifies that each distinguishable
// outcome yields a complete, internally-consistent verdict record and that
// no non-success outcome can masquerade as a success.
func TestReportCompleteForEveryOutcome(t *testing.T) {
	n := 40
	Asym := laplacianSym(n)
	Ag := laplacianGeneral(n)
	b := make([]float64, n)
	for i := range b {
		b[i] = 1
	}

	t.Run("converged", func(t *testing.T) {
		r, err := Solve(context.Background(), Asym, b, Config{Tolerance: 1e-9, Method: MethodCG})
		if err != nil {
			t.Fatal(err)
		}
		rep := r.Report()
		if rep.Status != "converged" || !rep.Success {
			t.Fatalf("rep=%+v", rep)
		}
		if !rep.ResidualMeasured || !rep.ResidualMet {
			t.Fatalf("converged record must have measured, met residual: %+v", rep)
		}
		if rep.Iterations != r.Iterations || rep.FinalRelRes != r.FinalRelRes {
			t.Fatal("record numbers disagree with result")
		}
		if len(rep.History) != len(r.History) || rep.Tolerance != 1e-9 {
			t.Fatal("history/tolerance missing from record")
		}
		if rep.Basis == "" || rep.Method != "cg" {
			t.Fatalf("basis/method missing: %+v", rep)
		}
	})

	t.Run("max_iterations", func(t *testing.T) {
		r, _ := Solve(context.Background(), Asym, b, Config{Tolerance: 1e-15, MaxIterations: 2, Method: MethodCG})
		rep := r.Report()
		if rep.Status != "max_iterations" || rep.Success {
			t.Fatalf("rep=%+v", rep)
		}
		if rep.Iterations != 2 || !rep.ResidualMeasured || rep.ResidualMet {
			t.Fatalf("incomplete max_iterations record: %+v", rep)
		}
		if rep.FinalRelRes <= rep.Tolerance {
			t.Fatal("residual met despite max iterations")
		}
		if rep.Basis == "" {
			t.Fatal("missing basis")
		}
	})

	t.Run("breakdown", func(t *testing.T) {
		indef, _ := (&sparse.COO{Rows: 2, Cols: 2, Symmetric: true, Triplets: []sparse.Triplet{
			{Row: 0, Col: 0, Value: 1}, {Row: 1, Col: 0, Value: 2}, {Row: 1, Col: 1, Value: -4},
		}}).ToCSR()
		r, _ := Solve(context.Background(), indef, []float64{1, 1}, Config{Method: MethodCG, Tolerance: 1e-8})
		if r.Status != StatusBreakdown {
			t.Fatalf("status=%s", r.Status)
		}
		rep := r.Report()
		if rep.Status != "breakdown" || rep.Success || rep.ResidualMet {
			t.Fatalf("rep=%+v", rep)
		}
		if rep.Detail == "" || rep.Basis == "" {
			t.Fatal("breakdown record must explain itself")
		}
	})

	t.Run("invalid_input", func(t *testing.T) {
		r, err := Solve(context.Background(), Asym, []float64{1, 2}, Config{Tolerance: 1e-9})
		var ie *sparse.InputError
		if !errors.As(err, &ie) || ie.Kind != sparse.KindStructure {
			t.Fatalf("err=%v", err)
		}
		rep := r.Report()
		if rep.Status != "invalid_input" || rep.Success || rep.ResidualMeasured || rep.ResidualMet {
			t.Fatalf("rep=%+v", rep)
		}
		if rep.Detail == "" || rep.Basis == "" {
			t.Fatal("invalid-input record must explain itself")
		}
	})

	t.Run("canceled", func(t *testing.T) {
		ctx, cancel := context.WithCancel(context.Background())
		go func() {
			time.Sleep(2 * time.Millisecond)
			cancel()
		}()
		r, _ := Solve(ctx, Ag, b, Config{Method: MethodGMRES, Tolerance: 1e-15, MaxIterations: 100000, Restart: n})
		if r.Status != StatusCanceled {
			t.Skipf("solver finished before cancel: %s", r.Status)
		}
		rep := r.Report()
		if rep.Status != "canceled" || rep.Success {
			t.Fatalf("rep=%+v", rep)
		}
		if rep.History == nil || len(rep.History) == 0 {
			t.Fatal("canceled record must carry the convergence trajectory so far")
		}
		if rep.Iterations != r.Iterations {
			t.Fatal("canceled record must report steps actually taken")
		}
		if !rep.ResidualMeasured {
			t.Fatal("canceled segment reports an explicitly measured residual")
		}
		if rep.Basis == "" || rep.Detail == "" {
			t.Fatal("canceled record must explain why and where it stopped")
		}
	})

	t.Run("paused", func(t *testing.T) {
		r, _ := Advance(context.Background(), Asym, b,
			Config{Tolerance: 1e-12, MaxIterations: 100, Method: MethodCG}, nil, 3)
		if r.Status != StatusPaused {
			t.Fatalf("status=%s", r.Status)
		}
		rep := r.Report()
		if rep.Status != "paused" || rep.Success || rep.ResidualMet {
			t.Fatalf("rep=%+v", rep)
		}
		if rep.SegmentIterations != 3 || rep.Iterations != 3 {
			t.Fatalf("segment accounting wrong: %+v", rep)
		}
		if !rep.ResidualMeasured || rep.FinalRelRes <= rep.Tolerance {
			t.Fatal("paused record residual wrong")
		}
	})
}

// TestReportJSONRoundTrip ensures records are directly archivable.
func TestReportJSONRoundTrip(t *testing.T) {
	r, _ := Solve(context.Background(), laplacianSym(10),
		make([]float64, 10), Config{Tolerance: 1e-8, Method: MethodCG})
	r.X = nil // the report must not need X
	raw, err := json.Marshal(r.Report())
	if err != nil {
		t.Fatal(err)
	}
	var rep Report
	if err := json.Unmarshal(raw, &rep); err != nil {
		t.Fatal(err)
	}
	if rep.Status != "converged" || !rep.Success || math.IsNaN(rep.FinalRelRes) {
		t.Fatalf("archived record unusable: %s", raw)
	}
}

// TestNeverConvergedAboveTolerance is a cross-outcome invariant check.
func TestNeverConvergedAboveTolerance(t *testing.T) {
	cfgs := []Config{
		{Tolerance: 1e-15, MaxIterations: 1, Method: MethodCG},
		{Tolerance: 1e-15, MaxIterations: 1, Method: MethodGMRES, Restart: 5},
	}
	for _, cfg := range cfgs {
		r, _ := Solve(context.Background(), laplacianSym(12), make([]float64, 12), cfg)
		if r.Status == StatusConverged && r.FinalRelRes > cfg.Tolerance {
			t.Fatalf("%v reported converged with rel %.3e > tol", cfg.Method, r.FinalRelRes)
		}
		if rep := r.Report(); rep.Success && !rep.ResidualMet {
			t.Fatal("archived record marked success without meeting residual")
		}
	}
}
