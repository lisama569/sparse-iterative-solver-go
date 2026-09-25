// Package sparse 提供稀疏矩阵的存储、校验与本地格式加载。
package sparse

import (
	"errors"
	"fmt"
	"math"
	"sort"
)

// 错误分类：格式错误、结构冲突、数值不可用，三者可区分。
var (
	ErrFormat     = errors.New("sparse: 格式错误")
	ErrStructure  = errors.New("sparse: 结构冲突")
	ErrBadValue   = errors.New("sparse: 数值不可用")
	ErrDimension  = errors.New("sparse: 维度不一致")
	ErrDuplicate  = errors.New("sparse: 重复的矩阵项")
	ErrBadIndex   = errors.New("sparse: 非法索引")
	ErrNotSquare  = errors.New("sparse: 矩阵不是方阵")
	ErrAsymmetric = errors.New("sparse: 矩阵不满足结构对称")
)

// Matrix 是不可变的 CSR 稀疏矩阵。构建完成后不得修改。
type Matrix struct {
	Rows, Cols int
	RowPtr     []int // 长度 Rows+1
	ColIdx     []int // 长度 nnz
	Vals       []float64
}

// NNZ 返回非零元个数。
func (m *Matrix) NNZ() int { return len(m.Vals) }

// Entry 描述一个坐标三元组（0-based）。
type Entry struct {
	Row, Col int
	Val      float64
}

// BuildFromCOO 由三元组构建 CSR 矩阵。
// 校验：维度必须为正；索引必须在界内；数值必须有限；
// 重复的 (row,col) 视为结构冲突，直接报错，不做合并。
func BuildFromCOO(rows, cols int, entries []Entry) (*Matrix, error) {
	if rows <= 0 || cols <= 0 {
		return nil, fmt.Errorf("%w: 维度必须为正, got %d x %d", ErrDimension, rows, cols)
	}
	sorted := make([]Entry, len(entries))
	copy(sorted, entries)
	for i, e := range sorted {
		if e.Row < 0 || e.Row >= rows || e.Col < 0 || e.Col >= cols {
			return nil, fmt.Errorf("%w: 第 %d 项 (%d,%d) 超出 %d x %d",
				ErrBadIndex, i, e.Row, e.Col, rows, cols)
		}
		if math.IsNaN(e.Val) || math.IsInf(e.Val, 0) {
			return nil, fmt.Errorf("%w: 第 %d 项 (%d,%d) 的值为 %v",
				ErrBadValue, i, e.Row, e.Col, e.Val)
		}
	}
	sort.SliceStable(sorted, func(i, j int) bool {
		if sorted[i].Row != sorted[j].Row {
			return sorted[i].Row < sorted[j].Row
		}
		return sorted[i].Col < sorted[j].Col
	})
	for i := 1; i < len(sorted); i++ {
		if sorted[i].Row == sorted[i-1].Row && sorted[i].Col == sorted[i-1].Col {
			return nil, fmt.Errorf("%w: 位置 (%d,%d) 出现多次",
				ErrDuplicate, sorted[i].Row, sorted[i].Col)
		}
	}

	m := &Matrix{
		Rows:   rows,
		Cols:   cols,
		RowPtr: make([]int, rows+1),
		ColIdx: make([]int, len(sorted)),
		Vals:   make([]float64, len(sorted)),
	}
	for _, e := range sorted {
		m.RowPtr[e.Row+1]++
	}
	for i := 0; i < rows; i++ {
		m.RowPtr[i+1] += m.RowPtr[i]
	}
	for i, e := range sorted {
		m.ColIdx[i] = e.Col
		m.Vals[i] = e.Val
	}
	return m, nil
}

// MatVec 计算 y = A*x。x 长度必须等于列数。
func (m *Matrix) MatVec(x []float64) ([]float64, error) {
	if len(x) != m.Cols {
		return nil, fmt.Errorf("%w: A 是 %d x %d, x 长度 %d",
			ErrDimension, m.Rows, m.Cols, len(x))
	}
	y := make([]float64, m.Rows)
	for i := 0; i < m.Rows; i++ {
		var s float64
		for k := m.RowPtr[i]; k < m.RowPtr[i+1]; k++ {
			s += m.Vals[k] * x[m.ColIdx[k]]
		}
		y[i] = s
	}
	return y, nil
}

// Diagonal 返回对角线；要求方阵且每个对角元存在且非零。
func (m *Matrix) Diagonal() ([]float64, error) {
	if m.Rows != m.Cols {
		return nil, fmt.Errorf("%w: %d x %d", ErrNotSquare, m.Rows, m.Cols)
	}
	d := make([]float64, m.Rows)
	for i := 0; i < m.Rows; i++ {
		found := false
		for k := m.RowPtr[i]; k < m.RowPtr[i+1]; k++ {
			if m.ColIdx[k] == i {
				if m.Vals[k] == 0 {
					return nil, fmt.Errorf("%w: 对角元 (%d,%d) 为零", ErrBadValue, i, i)
				}
				d[i] = m.Vals[k]
				found = true
				break
			}
		}
		if !found {
			return nil, fmt.Errorf("%w: 缺少对角元 (%d,%d)", ErrStructure, i, i)
		}
	}
	return d, nil
}

// CheckSymmetricStructure 校验方阵且每个 (i,j) 都有对应 (j,i)。
func (m *Matrix) CheckSymmetricStructure() error {
	if m.Rows != m.Cols {
		return fmt.Errorf("%w: %d x %d", ErrNotSquare, m.Rows, m.Cols)
	}
	// 用列索引集合做存在性检查。
	for i := 0; i < m.Rows; i++ {
		for k := m.RowPtr[i]; k < m.RowPtr[i+1]; k++ {
			j := m.ColIdx[k]
			if j == i {
				continue
			}
			if !m.hasEntry(j, i) {
				return fmt.Errorf("%w: 存在 (%d,%d) 但缺少 (%d,%d)",
					ErrAsymmetric, i, j, j, i)
			}
		}
	}
	return nil
}

func (m *Matrix) hasEntry(row, col int) bool {
	lo, hi := m.RowPtr[row], m.RowPtr[row+1]
	// 每行 ColIdx 有序，二分查找。
	for lo < hi {
		mid := (lo + hi) / 2
		switch {
		case m.ColIdx[mid] < col:
			lo = mid + 1
		case m.ColIdx[mid] > col:
			hi = mid
		default:
			return true
		}
	}
	return false
}
