package sparse

import (
	"math"
	"sort"
)

// COO is a coordinate (triplet) sparse matrix. A COO that has been
// converted successfully with ToCSR is treated as immutable input, which
// lets concurrent solvers share it without copying.
type COO struct {
	Rows, Cols int
	Triplets   []Triplet
	Symmetric  bool
}

// Triplet is a single (row, col, value) coordinate. Indices are zero-based.
type Triplet struct {
	Row, Col int
	Value    float64
}

// CSR is an immutable compressed sparse row matrix. Column indices within
// each row are sorted ascending.
type CSR struct {
	Rows, Cols int
	RowPtr     []int
	ColInd     []int
	Values     []float64
	Symmetric  bool
}

// Dims returns the matrix dimensions.
func (c *COO) Dims() (rows, cols int) { return c.Rows, c.Cols }

// Dims returns the matrix dimensions.
func (m *CSR) Dims() (rows, cols int) { return m.Rows, m.Cols }

// NNZ returns the number of stored entries.
func (m *CSR) NNZ() int {
	if len(m.RowPtr) == 0 {
		return 0
	}
	return m.RowPtr[m.Rows]
}

// ToCSR validates the coordinate matrix and converts it to immutable CSR.
//
// The following conditions are rejected rather than silently repaired:
//   - non-positive dimensions (structure error),
//   - row/column indices outside the declared dimensions (structure error),
//   - duplicate coordinates (structure error); for symmetric matrices a
//     pair (i,j)/(j,i) supplied together also counts as a duplicate,
//   - NaN or +-Inf values (numeric error).
//
// Symmetric matrices must be square and are expanded to full storage; only
// one triangle (plus optionally the diagonal) is accepted on input.
func (c *COO) ToCSR() (*CSR, error) {
	if c == nil {
		return nil, newError(KindStructure, "nil matrix")
	}
	if c.Rows <= 0 || c.Cols <= 0 {
		return nil, newError(KindStructure, "invalid dimensions %dx%d", c.Rows, c.Cols)
	}
	if c.Symmetric && c.Rows != c.Cols {
		return nil, newError(KindStructure, "symmetric matrix must be square, got %dx%d", c.Rows, c.Cols)
	}
	type key struct{ r, c int }
	seen := make(map[key]int, len(c.Triplets)) // key -> index for messages
	expanded := make([]Triplet, 0, len(c.Triplets)*2)
	for ti, t := range c.Triplets {
		if t.Row < 0 || t.Row >= c.Rows || t.Col < 0 || t.Col >= c.Cols {
			return nil, newError(KindStructure,
				"triplet %d has index (%d,%d) outside %dx%d", ti, t.Row, t.Col, c.Rows, c.Cols)
		}
		if math.IsNaN(t.Value) || math.IsInf(t.Value, 0) {
			return nil, newError(KindNumeric,
				"triplet %d at (%d,%d) has non-finite value %v", ti, t.Row, t.Col, t.Value)
		}
		k := key{t.Row, t.Col}
		if c.Symmetric {
			// Canonicalize on the unordered pair so (i,j) and (j,i) collide.
			if t.Col < t.Row {
				k = key{t.Col, t.Row}
			}
		}
		if prev, ok := seen[k]; ok {
			return nil, newError(KindStructure,
				"duplicate entry at (%d,%d): triplets %d and %d", t.Row, t.Col, prev, ti)
		}
		seen[k] = ti
		expanded = append(expanded, t)
		if c.Symmetric && t.Row != t.Col {
			expanded = append(expanded, Triplet{Row: t.Col, Col: t.Row, Value: t.Value})
		}
	}

	// Assemble CSR.
	rowCount := make([]int, c.Rows+1)
	for _, t := range expanded {
		rowCount[t.Row+1]++
	}
	rowPtr := make([]int, c.Rows+1)
	for i := 1; i <= c.Rows; i++ {
		rowPtr[i] = rowPtr[i-1] + rowCount[i]
	}
	nnz := rowPtr[c.Rows]
	colInd := make([]int, nnz)
	vals := make([]float64, nnz)
	cursor := make([]int, c.Rows)
	copy(cursor, rowPtr[:c.Rows])
	for _, t := range expanded {
		pos := cursor[t.Row]
		colInd[pos] = t.Col
		vals[pos] = t.Value
		cursor[t.Row]++
	}
	// Sort each row by column index for deterministic storage, keeping
	// values paired with their column indices.
	for i := 0; i < c.Rows; i++ {
		s, e := rowPtr[i], rowPtr[i+1]
		items := make([]rowItem, 0, e-s)
		for k := s; k < e; k++ {
			items = append(items, rowItem{colInd[k], vals[k]})
		}
		sort.Slice(items, func(a, b int) bool { return items[a].col < items[b].col })
		for j, it := range items {
			colInd[s+j] = it.col
			vals[s+j] = it.val
		}
	}
	m := &CSR{Rows: c.Rows, Cols: c.Cols, RowPtr: rowPtr, ColInd: colInd, Values: vals, Symmetric: c.Symmetric}
	return m, nil
}

// rowItem pairs a column with its value for stable per-row sorting.
type rowItem struct {
	col int
	val float64
}

// MatVec computes y = A*x. The input slice is not retained or modified.
func (m *CSR) MatVec(x []float64) ([]float64, error) {
	if m == nil {
		return nil, newError(KindStructure, "nil matrix")
	}
	if len(x) != m.Cols {
		return nil, newError(KindStructure,
			"operand length %d does not match matrix columns %d", len(x), m.Cols)
	}
	y := make([]float64, m.Rows)
	for i := 0; i < m.Rows; i++ {
		s := 0.0
		for k := m.RowPtr[i]; k < m.RowPtr[i+1]; k++ {
			s += m.Values[k] * x[m.ColInd[k]]
		}
		y[i] = s
	}
	return y, nil
}

// Diagonal returns a copy of the matrix diagonal.
func (m *CSR) Diagonal() []float64 {
	d := make([]float64, m.Rows)
	for i := 0; i < m.Rows; i++ {
		d[i] = 0
		for k := m.RowPtr[i]; k < m.RowPtr[i+1]; k++ {
			if m.ColInd[k] == i {
				d[i] = m.Values[k]
				break
			}
		}
	}
	return d
}
