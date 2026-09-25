package sparse

import (
	"errors"
	"math"
	"strings"
	"testing"
)

func TestBuildFromCOO_Valid(t *testing.T) {
	m, err := BuildFromCOO(2, 2, []Entry{
		{0, 0, 4}, {0, 1, 1}, {1, 0, 1}, {1, 1, 3},
	})
	if err != nil {
		t.Fatalf("构建失败: %v", err)
	}
	y, err := m.MatVec([]float64{1, 2})
	if err != nil {
		t.Fatalf("MatVec 失败: %v", err)
	}
	if y[0] != 6 || y[1] != 7 {
		t.Fatalf("MatVec 结果错误: %v", y)
	}
}

func TestBuildFromCOO_Errors(t *testing.T) {
	cases := []struct {
		name    string
		rows    int
		cols    int
		entries []Entry
		want    error
	}{
		{"零维度", 0, 3, nil, ErrDimension},
		{"行越界", 2, 2, []Entry{{2, 0, 1}}, ErrBadIndex},
		{"列越界", 2, 2, []Entry{{0, -1, 1}}, ErrBadIndex},
		{"NaN", 2, 2, []Entry{{0, 0, math.NaN()}}, ErrBadValue},
		{"Inf", 2, 2, []Entry{{0, 0, math.Inf(1)}}, ErrBadValue},
		{"重复项", 2, 2, []Entry{{0, 0, 1}, {0, 0, 2}}, ErrDuplicate},
	}
	for _, c := range cases {
		_, err := BuildFromCOO(c.rows, c.cols, c.entries)
		if !errors.Is(err, c.want) {
			t.Errorf("%s: 期望错误 %v, 得到 %v", c.name, c.want, err)
		}
	}
}

func TestMatVec_DimensionMismatch(t *testing.T) {
	m, _ := BuildFromCOO(2, 3, []Entry{{0, 0, 1}})
	if _, err := m.MatVec([]float64{1, 2}); !errors.Is(err, ErrDimension) {
		t.Fatalf("期望 ErrDimension, 得到 %v", err)
	}
}

func TestDiagonal(t *testing.T) {
	m, _ := BuildFromCOO(2, 2, []Entry{{0, 0, 4}, {1, 1, 3}})
	d, err := m.Diagonal()
	if err != nil || d[0] != 4 || d[1] != 3 {
		t.Fatalf("Diagonal = %v, err = %v", d, err)
	}
	// 缺对角元
	m2, _ := BuildFromCOO(2, 2, []Entry{{0, 0, 4}, {0, 1, 1}})
	if _, err := m2.Diagonal(); !errors.Is(err, ErrStructure) {
		t.Fatalf("期望 ErrStructure, 得到 %v", err)
	}
	// 零对角元
	m3, _ := BuildFromCOO(2, 2, []Entry{{0, 0, 0}, {1, 1, 3}})
	if _, err := m3.Diagonal(); !errors.Is(err, ErrBadValue) {
		t.Fatalf("期望 ErrBadValue, 得到 %v", err)
	}
	// 非方阵
	m4, _ := BuildFromCOO(2, 3, []Entry{{0, 0, 1}})
	if _, err := m4.Diagonal(); !errors.Is(err, ErrNotSquare) {
		t.Fatalf("期望 ErrNotSquare, 得到 %v", err)
	}
}

func TestCheckSymmetricStructure(t *testing.T) {
	sym, _ := BuildFromCOO(2, 2, []Entry{{0, 0, 2}, {0, 1, 1}, {1, 0, 1}, {1, 1, 2}})
	if err := sym.CheckSymmetricStructure(); err != nil {
		t.Fatalf("对称矩阵校验失败: %v", err)
	}
	asym, _ := BuildFromCOO(2, 2, []Entry{{0, 0, 2}, {0, 1, 1}, {1, 1, 2}})
	if err := asym.CheckSymmetricStructure(); !errors.Is(err, ErrAsymmetric) {
		t.Fatalf("期望 ErrAsymmetric, 得到 %v", err)
	}
}

const mmGeneral = `%%MatrixMarket matrix coordinate real general
% 注释行
3 3 5
1 1 4.0
1 2 1.0
2 2 3.0
3 1 2.0
3 3 1.0
`

func TestLoadMatrixMarket_General(t *testing.T) {
	m, err := LoadMatrixMarket(strings.NewReader(mmGeneral))
	if err != nil {
		t.Fatalf("解析失败: %v", err)
	}
	if m.Rows != 3 || m.Cols != 3 || m.NNZ() != 5 {
		t.Fatalf("尺寸错误: %d x %d nnz=%d", m.Rows, m.Cols, m.NNZ())
	}
}

func TestLoadMatrixMarket_SymmetricExpands(t *testing.T) {
	src := `%%MatrixMarket matrix coordinate real symmetric
2 2 3
1 1 2.0
2 1 1.0
2 2 2.0
`
	m, err := LoadMatrixMarket(strings.NewReader(src))
	if err != nil {
		t.Fatalf("解析失败: %v", err)
	}
	if m.NNZ() != 4 {
		t.Fatalf("对称展开后应有 4 个非零元, 得到 %d", m.NNZ())
	}
	if err := m.CheckSymmetricStructure(); err != nil {
		t.Fatalf("展开后应结构对称: %v", err)
	}
}

func TestLoadMatrixMarket_Errors(t *testing.T) {
	cases := []struct {
		name string
		src  string
		want error
	}{
		{"空输入", "", ErrFormat},
		{"坏头部", "not a header\n1 1 1\n1 1 1\n", ErrFormat},
		{"不支持的类型", "%%MatrixMarket matrix coordinate complex general\n1 1 1\n1 1 1\n", ErrFormat},
		{"不支持的对称性", "%%MatrixMarket matrix coordinate real hermitian\n1 1 1\n1 1 1\n", ErrFormat},
		{"索引越界", "%%MatrixMarket matrix coordinate real general\n2 2 1\n3 1 1.0\n", ErrBadIndex},
		{"数量不符", "%%MatrixMarket matrix coordinate real general\n2 2 2\n1 1 1.0\n", ErrFormat},
		{"重复项", "%%MatrixMarket matrix coordinate real general\n2 2 2\n1 1 1.0\n1 1 2.0\n", ErrDuplicate},
		{"NaN 值", "%%MatrixMarket matrix coordinate real general\n2 2 1\n1 1 NaN\n", ErrBadValue},
		{"对称展开冲突", "%%MatrixMarket matrix coordinate real symmetric\n2 2 2\n2 1 1.0\n1 2 2.0\n", ErrDuplicate},
	}
	for _, c := range cases {
		_, err := LoadMatrixMarket(strings.NewReader(c.src))
		if !errors.Is(err, c.want) {
			t.Errorf("%s: 期望 %v, 得到 %v", c.name, c.want, err)
		}
	}
}

func TestLoadCOO(t *testing.T) {
	src := `# 测试
3 3 3
1 1 2.0
2 2 3.0
3 3 4.0
`
	m, err := LoadCOO(strings.NewReader(src))
	if err != nil {
		t.Fatalf("解析失败: %v", err)
	}
	d, err := m.Diagonal()
	if err != nil || d[0] != 2 || d[1] != 3 || d[2] != 4 {
		t.Fatalf("对角线 = %v, err = %v", d, err)
	}
}

func TestLoadCOO_Errors(t *testing.T) {
	cases := []struct {
		name string
		src  string
		want error
	}{
		{"空输入", "", ErrFormat},
		{"坏尺寸行", "2 2\n", ErrFormat},
		{"索引越界", "2 2 1\n0 1 1.0\n", ErrBadIndex},
		{"数量不符", "2 2 2\n1 1 1.0\n", ErrFormat},
		{"重复项", "2 2 2\n1 1 1.0\n1 1 2.0\n", ErrDuplicate},
	}
	for _, c := range cases {
		_, err := LoadCOO(strings.NewReader(c.src))
		if !errors.Is(err, c.want) {
			t.Errorf("%s: 期望 %v, 得到 %v", c.name, c.want, err)
		}
	}
}
