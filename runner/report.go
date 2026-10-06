package runner

import "github.com/lisama569/sparse-iterative-solver-go/solver"

// Record returns the archivable verdict record of a batch task, stamped
// with the task name. It is safe to embed in a batch summary or JSON log.
func (tr TaskResult) Record() solver.Report {
	if tr.Result == nil {
		return solver.Report{
			Name:    tr.Name,
			Status:  solver.StatusBreakdown.String(),
			Basis:   "batch produced no result for this task",
			Detail:  "missing result",
			History: []float64{},
		}
	}
	rep := tr.Result.Report()
	rep.Name = tr.Name
	return rep
}

// Records returns one archivable record per task, in submission order.
func (s *BatchSummary) Records() []solver.Report {
	out := make([]solver.Report, len(s.Results))
	for i := range s.Results {
		out[i] = s.Results[i].Record()
	}
	return out
}
