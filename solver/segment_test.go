package solver

import (
	"context"
	"math"
	"testing"

	"github.com/lisama569/sparse-iterative-solver-go/sparse"
)

// advanceSegments solves by repeated Advance calls with the given segment
// caps. After the listed caps run out, the last cap is repeated until a
// terminal conclusion. It returns the last result and every result in order.
func advanceSegments(t *testing.T, A *sparse.CSR, b []float64, cfg Config, segCaps []int) (*Result, []*Result) {
	t.Helper()
	var prog *Progress
	var all []*Result
	for k := 0; k < 100000; k++ {
		cap := segCaps[0]
		if k < len(segCaps) {
			cap = segCaps[k]
		}
		r, err := Advance(context.Background(), A, b, cfg, prog, cap)
		if err != nil && r.Status != StatusInvalidInput {
			t.Fatalf("Advance err: %v", err)
		}
		all = append(all, r)
		if r.Status.Terminal() {
			return r, all
		}
		if r.Progress == nil {
			t.Fatalf("non-terminal status %s without progress", r.Status)
		}
		prog = r.Progress
	}
	t.Fatal("segmented solve never terminated")
	return nil, all
}

func TestSegmentedEqualsOneShotCG(t *testing.T) {
	n := 60
	A := laplacianSym(n)
	b := make([]float64, n)
	for i := range b {
		b[i] = 1
	}
	cfg := Config{Tolerance: 1e-12, MaxIterations: 500, Method: MethodCG}

	oneShot, err := Solve(context.Background(), A, b, cfg)
	if err != nil || oneShot.Status != StatusConverged {
		t.Fatalf("one-shot: status=%s err=%v", oneShot.Status, err)
	}

	for _, caps := range [][]int{
		{1}, {2}, {3}, {7}, {100}, {1, 1, 1}, {3, 1, 4, 1, 9, 1000},
	} {
		t.Run("", func(t *testing.T) {
			seg, parts := advanceSegments(t, A, b, cfg, caps)
			if seg.Status != StatusConverged {
				t.Fatalf("segmented caps=%v status=%s msg=%s", caps, seg.Status, seg.ErrorMessage)
			}
			if seg.Iterations != oneShot.Iterations {
				t.Fatalf("iters segmented=%d one-shot=%d", seg.Iterations, oneShot.Iterations)
			}
			if seg.FinalRelRes != oneShot.FinalRelRes {
				t.Fatalf("final residual segmented=%.17g one-shot=%.17g", seg.FinalRelRes, oneShot.FinalRelRes)
			}
			if len(seg.History) != len(oneShot.History) {
				t.Fatalf("history length segmented=%d one-shot=%d", len(seg.History), len(oneShot.History))
			}
			for i := range seg.History {
				if seg.History[i] != oneShot.History[i] {
					t.Fatalf("history[%d] segmented=%.17g one-shot=%.17g", i, seg.History[i], oneShot.History[i])
				}
			}
			for i := range seg.X {
				if seg.X[i] != oneShot.X[i] {
					t.Fatalf("x[%d] differs: %.17g vs %.17g", i, seg.X[i], oneShot.X[i])
				}
			}
			// Every non-terminal part must be paused with cumulative
			// history and a progress that matches reported iterations at
			// the moment it is handed back.
			for i, p := range parts {
				if p.Status.Terminal() {
					continue
				}
				if p.Status != StatusPaused {
					t.Fatalf("part %d status=%s want paused", i, p.Status)
				}
				progIters := -1
				var progRes float64
				if p.Progress != nil {
					progIters = p.Progress.Iterations()
					progRes = p.Progress.Residual()
				}
				if p.Progress == nil || progIters != p.Iterations || progRes != p.FinalRelRes {
					t.Fatalf("part %d progress mismatch: iters=%d/%d res=%.17g/%.17g",
						i, progIters, p.Iterations, progRes, p.FinalRelRes)
				}
				if p.FinalRelRes <= cfg.Tolerance {
					t.Fatal("paused result must not be within tolerance")
				}
			}
			// Segment iteration counts must sum to the total.
			var segSum int
			for _, p := range parts {
				segSum += p.SegmentIterations
			}
			if segSum != seg.Iterations {
				t.Fatalf("segment steps sum=%d != total iters=%d", segSum, seg.Iterations)
			}
		})
	}
}

func TestSegmentedEqualsOneShotGMRES(t *testing.T) {
	n := 40
	A := laplacianGeneral(n)
	b := make([]float64, n)
	for i := range b {
		b[i] = float64(i%5 - 2)
	}
	cfg := Config{Tolerance: 1e-10, MaxIterations: 2000, Method: MethodGMRES, Restart: 12}

	oneShot, err := Solve(context.Background(), A, b, cfg)
	if err != nil || oneShot.Status != StatusConverged {
		t.Fatalf("one-shot: status=%s err=%v msg=%s", oneShot.Status, err, oneShot.ErrorMessage)
	}
	for _, caps := range [][]int{{1}, {5}, {12}, {13}, {25}, {3, 7, 2, 50}} {
		t.Run("", func(t *testing.T) {
			seg, _ := advanceSegments(t, A, b, cfg, caps)
			if seg.Status != StatusConverged {
				t.Fatalf("segmented caps=%v status=%s msg=%s", caps, seg.Status, seg.ErrorMessage)
			}
			if seg.Iterations != oneShot.Iterations {
				t.Fatalf("iters segmented=%d one-shot=%d", seg.Iterations, oneShot.Iterations)
			}
			if math.Abs(seg.FinalRelRes-oneShot.FinalRelRes) > 1e-14*math.Max(1, oneShot.FinalRelRes) {
				t.Fatalf("final residual segmented=%.17g one-shot=%.17g", seg.FinalRelRes, oneShot.FinalRelRes)
			}
			checkResidual(t, A, seg.X, b, cfg.Tolerance)
		})
	}
}

// TestSegmentedMatchesKnownTrueSolution uses the tridiagonal Laplacian with
// a manufactured RHS from a known smooth x*.
func TestSegmentedMatchesKnownTrueSolution(t *testing.T) {
	n := 30
	A := laplacianSym(n)
	xStar := make([]float64, n)
	for i := range xStar {
		xStar[i] = math.Sin(float64(i+1) * math.Pi / float64(n+1))
	}
	Ax, _ := A.MatVec(xStar)
	b := Ax // manufactured RHS

	for _, m := range []struct {
		name string
		cfg  Config
	}{
		{"cg", Config{Tolerance: 1e-11, MaxIterations: 1000, Method: MethodCG}},
		{"gmres", Config{Tolerance: 1e-11, MaxIterations: 1000, Method: MethodGMRES, Restart: 20}},
		{"cg-prec", Config{Tolerance: 1e-11, MaxIterations: 1000, Method: MethodCG, Precondition: true}},
	} {
		t.Run(m.name, func(t *testing.T) {
			seg, _ := advanceSegments(t, A, b, m.cfg, []int{4, 7, 3, 1000})
			if seg.Status != StatusConverged {
				t.Fatalf("status=%s msg=%s", seg.Status, seg.ErrorMessage)
			}
			checkResidual(t, A, seg.X, b, m.cfg.Tolerance)
			var maxErr float64
			for i := range xStar {
				if d := math.Abs(seg.X[i] - xStar[i]); d > maxErr {
					maxErr = d
				}
			}
			if maxErr > 1e-8 {
				t.Fatalf("max error vs true solution %.3e", maxErr)
			}
		})
	}
}

// TestSegmentPauseThenMaxIterations: segmenting does not change the total
// budget; a paused run resumed with a tiny tolerance exhausts the same cap.
func TestSegmentPauseThenMaxIterations(t *testing.T) {
	A := laplacianSym(20)
	b := make([]float64, 20)
	for i := range b {
		b[i] = 1
	}
	cfg := Config{Tolerance: 1e-15, MaxIterations: 5, Method: MethodCG}
	oneShot, _ := Solve(context.Background(), A, b, cfg)
	seg, _ := advanceSegments(t, A, b, cfg, []int{2, 2, 2})
	if seg.Status != StatusMaxIterations {
		t.Fatalf("status=%s want max_iterations", seg.Status)
	}
	if seg.Iterations != oneShot.Iterations || seg.Iterations != 5 {
		t.Fatalf("iters seg=%d oneshot=%d", seg.Iterations, oneShot.Iterations)
	}
	if seg.FinalRelRes != oneShot.FinalRelRes {
		t.Fatalf("residual mismatch")
	}
}

// TestSegmentProgressMismatch verifies handoff errors never corrupt progress.
func TestSegmentProgressMismatch(t *testing.T) {
	A := laplacianSym(10)
	b := make([]float64, 10)
	for i := range b {
		b[i] = 1
	}
	cfg := Config{Tolerance: 1e-12, MaxIterations: 100, Method: MethodCG}
	r1, _ := Advance(context.Background(), A, b, cfg, nil, 3)
	if r1.Status != StatusPaused {
		t.Fatalf("status=%s", r1.Status)
	}
	snap := r1.Progress.Residual()
	snapIter := r1.Progress.Iterations()

	// Different tolerance.
	bad := cfg
	bad.Tolerance = 1e-6
	r2, err := Advance(context.Background(), A, b, bad, r1.Progress, 3)
	if err == nil || r2.Status != StatusInvalidInput {
		t.Fatalf("expected mismatch error, got %v status=%s", err, r2.Status)
	}
	// Progress is untouched.
	if r1.Progress.Iterations() != snapIter || r1.Progress.Residual() != snap {
		t.Fatal("progress was mutated by a rejected continuation")
	}
	// The same progress can still be continued correctly.
	r3, _ := Advance(context.Background(), A, b, cfg, r1.Progress, 1000)
	if r3.Status != StatusConverged {
		t.Fatalf("status=%s msg=%s", r3.Status, r3.ErrorMessage)
	}
}

// TestSegmentedCancellationResumable: cancel one segment, then resume.
func TestSegmentedCancellationResumable(t *testing.T) {
	n := 50
	A := laplacianSym(n)
	b := make([]float64, n)
	for i := range b {
		b[i] = 1
	}
	cfg := Config{Tolerance: 1e-13, MaxIterations: 1000, Method: MethodCG}

	ctx, cancel := context.WithCancel(context.Background())
	r1, _ := Advance(ctx, A, b, cfg, nil, 2)
	if r1.Status != StatusPaused {
		t.Fatalf("seg1 status=%s", r1.Status)
	}
	// Cancel before resuming: the segment must stop promptly but keep a
	// resumable checkpoint of everything previously earned.
	cancel()
	r2, _ := Advance(ctx, A, b, cfg, r1.Progress, 2)
	if r2.Status != StatusCanceled || r2.Progress == nil {
		t.Fatalf("seg2 status=%s prog=%v", r2.Status, r2.Progress)
	}
	if r2.Iterations < r1.Iterations {
		t.Fatal("canceled segment must not lose progress already earned")
	}
	r3, err := Advance(context.Background(), A, b, cfg, r2.Progress, 0)
	if err != nil {
		t.Fatal(err)
	}
	if r3.Status != StatusConverged {
		t.Fatalf("resume after cancel status=%s msg=%s", r3.Status, r3.ErrorMessage)
	}
	oneShot, _ := Solve(context.Background(), A, b, cfg)
	if r3.Iterations != oneShot.Iterations || r3.FinalRelRes != oneShot.FinalRelRes {
		t.Fatalf("resume-after-cancel mismatch: iters %d vs %d, res %.17g vs %.17g",
			r3.Iterations, oneShot.Iterations, r3.FinalRelRes, oneShot.FinalRelRes)
	}
}

func laplacianSym(n int) *sparse.CSR {
	var trips []sparse.Triplet
	for i := 0; i < n; i++ {
		trips = append(trips, sparse.Triplet{Row: i, Col: i, Value: 2})
		if i > 0 {
			trips = append(trips, sparse.Triplet{Row: i, Col: i - 1, Value: -1})
		}
	}
	m, err := (&sparse.COO{Rows: n, Cols: n, Symmetric: true, Triplets: trips}).ToCSR()
	if err != nil {
		panic(err)
	}
	return m
}
