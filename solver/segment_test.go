package solver

import (
	"context"
	"encoding/json"
	"math"
	"strings"
	"testing"

	"github.com/lisama569/sparse-iterative-solver-go/sparse"
)

// symTridiag builds the n x n SPD tridiagonal 4,1 matrix via the symmetric
// path so CG is exercised.
func symTridiag(n int) *sparse.CSR {
	var trips []sparse.Triplet
	for i := 0; i < n; i++ {
		trips = append(trips, sparse.Triplet{Row: i, Col: i, Value: 4})
		if i+1 < n {
			trips = append(trips, sparse.Triplet{Row: i, Col: i + 1, Value: 1})
		}
	}
	m, err := (&sparse.COO{Rows: n, Cols: n, Symmetric: true, Triplets: trips}).ToCSR()
	if err != nil {
		panic(err)
	}
	return m
}

// generalTridiag builds the n x n nonsymmetric-storage 4,1 tridiagonal so
// GMRES is exercised on a well-conditioned general matrix.
func generalTridiag(n int) *sparse.CSR {
	var trips []sparse.Triplet
	for i := 0; i < n; i++ {
		trips = append(trips, sparse.Triplet{Row: i, Col: i, Value: 4})
		if i > 0 {
			trips = append(trips, sparse.Triplet{Row: i, Col: i - 1, Value: 1})
			trips = append(trips, sparse.Triplet{Row: i - 1, Col: i, Value: 1})
		}
	}
	m, err := (&sparse.COO{Rows: n, Cols: n, Triplets: trips}).ToCSR()
	if err != nil {
		panic(err)
	}
	return m
}

// runSegments advances a solve in chunks of seg iterations until a terminal
// status, returning the final result.
func runSegments(t *testing.T, A *sparse.CSR, b []float64, cfg Config, seg int) *Result {
	t.Helper()
	var ck *Checkpoint
	for range 10000 {
		res, next, err := SolveSegment(context.Background(), A, b, cfg, ck, seg)
		if err != nil {
			t.Fatalf("SolveSegment: %v", err)
		}
		if res.Status == StatusPaused {
			if next == nil {
				t.Fatal("paused without a checkpoint")
			}
			ck = next
			continue
		}
		if next != nil {
			t.Fatalf("terminal status %s returned a checkpoint", res.Status)
		}
		return res
	}
	t.Fatal("solve did not terminate")
	return nil
}

// TestSegmentedMatchesOneShot verifies that splitting a solve into segments
// reaches the very same conclusion as one uninterrupted run: same status,
// same total iterations, same final residual, same trend, same iterate.
func TestSegmentedMatchesOneShot(t *testing.T) {
	n := 40
	b := make([]float64, n)
	for i := range b {
		b[i] = float64(i%7) + 1
	}
	cases := []struct {
		name string
		A    *sparse.CSR
		cfg  Config
	}{
		{"cg", symTridiag(n), Config{Method: MethodCG, Tolerance: 1e-11, MaxIterations: 300}},
		{"cg_precond", symTridiag(n), Config{Method: MethodCG, Tolerance: 1e-11, MaxIterations: 300, Precondition: true}},
		{"gmres", generalTridiag(n), Config{Method: MethodGMRES, Tolerance: 1e-11, MaxIterations: 300, Restart: 7}},
		{"gmres_precond", generalTridiag(n), Config{Method: MethodGMRES, Tolerance: 1e-11, MaxIterations: 300, Restart: 9, Precondition: true}},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			one, err := Solve(context.Background(), tc.A, b, tc.cfg)
			if err != nil {
				t.Fatal(err)
			}
			if one.Status != StatusConverged {
				t.Fatalf("one-shot did not converge: %s", one.Status)
			}
			// Segment sizes that do and do not align with the restart length.
			for _, seg := range []int{1, 5, 7, 13, 50} {
				segRes := runSegments(t, tc.A, b, tc.cfg, seg)
				if segRes.Status != one.Status {
					t.Fatalf("seg=%d: status %s != %s", seg, segRes.Status, one.Status)
				}
				if segRes.Iterations != one.Iterations {
					t.Fatalf("seg=%d: iterations %d != %d", seg, segRes.Iterations, one.Iterations)
				}
				if math.Abs(segRes.FinalRelRes-one.FinalRelRes) > 1e-14 {
					t.Fatalf("seg=%d: final relres %g != %g", seg, segRes.FinalRelRes, one.FinalRelRes)
				}
				if len(segRes.History) != len(one.History) {
					t.Fatalf("seg=%d: history length %d != %d", seg, len(segRes.History), len(one.History))
				}
				for i := range one.History {
					if math.Abs(segRes.History[i]-one.History[i]) > 1e-14 {
						t.Fatalf("seg=%d: history[%d] %g != %g", seg, i, segRes.History[i], one.History[i])
					}
				}
				for i := range one.X {
					if math.Abs(segRes.X[i]-one.X[i]) > 1e-12 {
						t.Fatalf("seg=%d: x[%d] %g != %g", seg, i, segRes.X[i], one.X[i])
					}
				}
			}
		})
	}
}

// TestSegmentedKnownSolution checks that a resumed solve both satisfies the
// residual requirement and recovers the true solution.
func TestSegmentedKnownSolution(t *testing.T) {
	n := 30
	A := symTridiag(n)
	xTrue := make([]float64, n)
	for i := range xTrue {
		xTrue[i] = math.Sin(float64(i)) + 0.5
	}
	b, err := A.MatVec(xTrue)
	if err != nil {
		t.Fatal(err)
	}
	cfg := Config{Method: MethodCG, Tolerance: 1e-10, MaxIterations: 500}
	res := runSegments(t, A, b, cfg, 4)
	if res.Status != StatusConverged {
		t.Fatalf("status %s", res.Status)
	}
	if res.FinalRelRes > cfg.Tolerance {
		t.Fatalf("final residual %g above tolerance %g", res.FinalRelRes, cfg.Tolerance)
	}
	// The measured residual of the returned iterate must genuinely meet the
	// tolerance (recomputed independently here).
	Ax, err := A.MatVec(res.X)
	if err != nil {
		t.Fatal(err)
	}
	r := make([]float64, n)
	for i := range r {
		r[i] = b[i] - Ax[i]
	}
	if got := safeRelNorm(r, vnorm(b)); got > cfg.Tolerance {
		t.Fatalf("independently measured residual %g above tolerance", got)
	}
	// Error against the known truth, scaled by the condition number of the
	// 4,1 tridiagonal (bounded by ~3 for any n).
	var maxErr float64
	for i := range res.X {
		maxErr = math.Max(maxErr, math.Abs(res.X[i]-xTrue[i]))
	}
	if maxErr > 1e-6 {
		t.Fatalf("solution error %g too large vs known truth", maxErr)
	}
}

// TestSegmentedMaxIterationsTerminal verifies that exhausting the total step
// budget across segments ends in a terminal max_iterations verdict with a
// complete record and no checkpoint.
func TestSegmentedMaxIterationsTerminal(t *testing.T) {
	n := 40
	A := symTridiag(n)
	b := make([]float64, n)
	for i := range b {
		b[i] = 1
	}
	cfg := Config{Method: MethodCG, Tolerance: 1e-14, MaxIterations: 11}
	res := runSegments(t, A, b, cfg, 3)
	if res.Status != StatusMaxIterations {
		t.Fatalf("status %s", res.Status)
	}
	if res.Iterations != 11 {
		t.Fatalf("iterations %d != 11", res.Iterations)
	}
	rec := res.Record()
	if rec.Converged {
		t.Fatal("max_iterations record must not be converged")
	}
	if rec.Iterations != 11 || rec.Reason == "" || len(rec.History) == 0 {
		t.Fatalf("incomplete record: %+v", rec)
	}
}

// TestSegmentCancelPreservesProgress cancels a segment mid-solve and checks
// that the checkpoint still resumes to the one-shot conclusion.
func TestSegmentCancelPreservesProgress(t *testing.T) {
	n := 40
	A := symTridiag(n)
	b := make([]float64, n)
	for i := range b {
		b[i] = float64(i%3) + 1
	}
	cfg := Config{Method: MethodCG, Tolerance: 1e-11, MaxIterations: 300}
	one, err := Solve(context.Background(), A, b, cfg)
	if err != nil {
		t.Fatal(err)
	}

	// First segment runs normally.
	_, ck, err := SolveSegment(context.Background(), A, b, cfg, nil, 5)
	if err != nil {
		t.Fatal(err)
	}
	if ck == nil || ck.Iterations == 0 {
		t.Fatalf("expected progress checkpoint, got %+v", ck)
	}
	// Second segment sees an already-canceled context: progress must survive.
	ctx, cancel := context.WithCancel(context.Background())
	cancel()
	res, ck2, err := SolveSegment(ctx, A, b, cfg, ck, 5)
	if err != nil {
		t.Fatal(err)
	}
	if res.Status != StatusCanceled {
		t.Fatalf("status %s", res.Status)
	}
	if ck2 == nil || ck2.Iterations != ck.Iterations {
		t.Fatalf("canceled segment lost progress: %+v", ck2)
	}
	// Resume with a live context: must finish exactly like the one-shot run.
	final := runSegments(t, A, b, cfg, 6)
	_ = final
	res, ck3, err := SolveSegment(context.Background(), A, b, cfg, ck2, 1000)
	if err != nil {
		t.Fatal(err)
	}
	if ck3 != nil {
		t.Fatal("expected terminal result")
	}
	if res.Status != one.Status || res.Iterations != one.Iterations {
		t.Fatalf("resumed conclusion %s/%d != one-shot %s/%d",
			res.Status, res.Iterations, one.Status, one.Iterations)
	}
	if math.Abs(res.FinalRelRes-one.FinalRelRes) > 1e-14 {
		t.Fatalf("residual %g != %g", res.FinalRelRes, one.FinalRelRes)
	}
}

// TestCheckpointMismatch verifies that a botched handoff is rejected and the
// saved progress is returned untouched.
func TestCheckpointMismatch(t *testing.T) {
	n := 20
	A := symTridiag(n)
	b := make([]float64, n)
	for i := range b {
		b[i] = 1
	}
	cfg := Config{Method: MethodCG, Tolerance: 1e-10, MaxIterations: 200}
	_, ck, err := SolveSegment(context.Background(), A, b, cfg, nil, 5)
	if err != nil {
		t.Fatal(err)
	}
	if ck == nil {
		t.Fatal("expected checkpoint")
	}

	bad := []Config{
		{Method: MethodCG, Tolerance: 1e-6, MaxIterations: 200},  // tolerance differs
		{Method: MethodCG, Tolerance: 1e-10, Precondition: true}, // preconditioner differs
	}
	for _, bc := range bad {
		res, back, err := SolveSegment(context.Background(), A, b, bc, ck, 5)
		if err == nil {
			t.Fatalf("cfg %+v: expected mismatch error", bc)
		}
		if res.Status != StatusInvalidInput {
			t.Fatalf("cfg %+v: status %s", bc, res.Status)
		}
		if back != ck {
			t.Fatalf("cfg %+v: saved checkpoint was not returned unchanged", bc)
		}
	}
	// Wrong dimension.
	big := symTridiag(n + 1)
	if _, _, err := SolveSegment(context.Background(), big, make([]float64, n+1), cfg, ck, 5); err == nil {
		t.Fatal("expected dimension mismatch error")
	}
	// Method mismatch: GMRES config against a CG checkpoint.
	gcfg := Config{Method: MethodGMRES, Tolerance: 1e-10, Restart: 5}
	if _, _, err := SolveSegment(context.Background(), A, b, gcfg, ck, 5); err == nil {
		t.Fatal("expected method mismatch error")
	}
	// The untouched checkpoint still resumes fine.
	res := runSegments(t, A, b, cfg, 1000)
	_ = res
	res2, _, err := SolveSegment(context.Background(), A, b, cfg, ck, 1000)
	if err != nil {
		t.Fatal(err)
	}
	if res2.Status != StatusConverged {
		t.Fatalf("status %s", res2.Status)
	}
}

// TestSegmentBudgetValidation rejects nonsensical segment budgets.
func TestSegmentBudgetValidation(t *testing.T) {
	A := spd3(t)
	b := []float64{1, 2, 3}
	if _, _, err := SolveSegment(context.Background(), A, b, Config{}, nil, -1); err == nil {
		t.Fatal("expected error for negative segment budget")
	}
}

// TestRecordVerdicts checks that every ending produces a complete,
// distinguishable archivable record.
func TestRecordVerdicts(t *testing.T) {
	A := spd3(t)
	b := []float64{1, 2, 3}

	converged, err := Solve(context.Background(), A, b, Config{Tolerance: 1e-10})
	if err != nil {
		t.Fatal(err)
	}
	maxIter, err := Solve(context.Background(), A, b, Config{Tolerance: 1e-16, MaxIterations: 1})
	if err != nil {
		t.Fatal(err)
	}
	indef, err := (&sparse.COO{Rows: 2, Cols: 2, Symmetric: true, Triplets: []sparse.Triplet{
		{Row: 0, Col: 0, Value: 1}, {Row: 1, Col: 1, Value: -1},
	}}).ToCSR()
	if err != nil {
		t.Fatal(err)
	}
	breakdown, err := Solve(context.Background(), indef, []float64{1, 1}, Config{Method: MethodCG})
	if err != nil {
		t.Fatal(err)
	}
	invalid, err := Solve(context.Background(), A, []float64{1, 2}, Config{})
	if err == nil {
		t.Fatal("expected invalid input error")
	}
	_, ckCancel, err := SolveSegment(context.Background(), A, b, Config{Tolerance: 1e-12}, nil, 1)
	if err != nil {
		t.Fatal(err)
	}
	ctx, cancel := context.WithCancel(context.Background())
	cancel()
	canceled, _, err := SolveSegment(ctx, A, b, Config{Tolerance: 1e-12}, ckCancel, 1)
	if err != nil {
		t.Fatal(err)
	}
	paused, ck, err := SolveSegment(context.Background(), A, b, Config{Tolerance: 1e-12}, nil, 1)
	if err != nil {
		t.Fatal(err)
	}
	if paused.Status != StatusPaused || ck == nil {
		t.Fatalf("expected paused with checkpoint, got %s", paused.Status)
	}
	skipped := &Result{Status: StatusSkipped, ErrorMessage: "batch budget exhausted"}

	results := map[string]*Result{
		"converged":      converged,
		"max_iterations": maxIter,
		"breakdown":      breakdown,
		"invalid_input":  invalid,
		"canceled":       canceled,
		"paused":         paused,
		"skipped":        skipped,
	}
	seen := map[string]bool{}
	for name, res := range results {
		rec := res.Record()
		if rec.Status == "" || rec.Status == "unknown" {
			t.Fatalf("%s: empty status", name)
		}
		if seen[rec.Status] {
			t.Fatalf("%s: duplicate status %s", name, rec.Status)
		}
		seen[rec.Status] = true
		if rec.Reason == "" {
			t.Fatalf("%s: empty reason", name)
		}
		if rec.Tolerance <= 0 && name != "skipped" {
			t.Fatalf("%s: tolerance %g not recorded", name, rec.Tolerance)
		}
		if name == "converged" {
			if !rec.Converged || rec.FinalRelRes > rec.Tolerance {
				t.Fatalf("converged record inconsistent: %+v", rec)
			}
		} else if rec.Converged {
			t.Fatalf("%s: record must not claim convergence", name)
		}
		if name != "invalid_input" && name != "skipped" && len(rec.History) == 0 {
			t.Fatalf("%s: history missing", name)
		}
		if _, err := json.Marshal(rec); err != nil {
			t.Fatalf("%s: record not serializable: %v", name, err)
		}
	}
	// Reasons must be distinguishable too.
	reasons := map[string]bool{}
	for _, res := range results {
		r := res.Record().Reason
		if reasons[r] {
			t.Fatalf("duplicate reason %q", r)
		}
		reasons[r] = true
	}
}

// TestRecordNeverFalseSuccess ensures a doctored "converged" flag without a
// measured residual meeting the tolerance is never reported as success.
func TestRecordNeverFalseSuccess(t *testing.T) {
	fake := &Result{Status: StatusConverged, FinalRelRes: 1.0, Tolerance: 1e-8, Iterations: 3}
	rec := fake.Record()
	if rec.Converged {
		t.Fatal("record claimed convergence without meeting the residual requirement")
	}
	if !strings.Contains(rec.Reason, "rejected") {
		t.Fatalf("reason should flag the inconsistency: %q", rec.Reason)
	}
	nan := &Result{Status: StatusConverged, FinalRelRes: math.NaN(), Tolerance: 1e-8}
	if nan.Record().Converged {
		t.Fatal("NaN residual must never converge")
	}
}
