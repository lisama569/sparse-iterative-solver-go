package mmio

import (
	"errors"
	"os"
	"path/filepath"
	"strings"
	"testing"

	"github.com/lisama569/sparse-iterative-solver-go/sparse"
)

func parse(t *testing.T, content string) *sparse.COO {
	t.Helper()
	c, err := ReadCOO(strings.NewReader(content))
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	return c
}

func expectKind(t *testing.T, content string, kind sparse.ErrorKind) {
	t.Helper()
	_, err := ReadCOO(strings.NewReader(content))
	var ie *sparse.InputError
	if !errors.As(err, &ie) {
		t.Fatalf("expected *InputError, got %v", err)
	}
	if ie.Kind != kind {
		t.Fatalf("kind = %d, want %d: %v", ie.Kind, kind, ie)
	}
}

func TestParseGeneral(t *testing.T) {
	src := `%%MatrixMarket matrix coordinate real general
% a comment
2 2 4
1 1 4.0
1 2 1.0
2 1 1.0
2 2 3.0
`
	c := parse(t, src)
	if c.Rows != 2 || c.Cols != 2 || len(c.Triplets) != 4 || c.Symmetric {
		t.Fatalf("unexpected COO: %+v", c)
	}
	m, err := c.ToCSR()
	if err != nil {
		t.Fatal(err)
	}
	y, _ := m.MatVec([]float64{1, 2})
	if y[0] != 6 || y[1] != 7 {
		t.Fatalf("matvec = %v", y)
	}
}

func TestParseSymmetricIntegerPattern(t *testing.T) {
	src := `%%MatrixMarket matrix coordinate integer symmetric
3 3 3
1 1 2
2 1 -1
3 3 2
`
	c := parse(t, src)
	if !c.Symmetric {
		t.Fatal("expected symmetric")
	}
	m, err := c.ToCSR()
	if err != nil {
		t.Fatal(err)
	}
	if m.NNZ() != 4 {
		t.Fatalf("nnz=%d want 4", m.NNZ())
	}

	pat := `%%MatrixMarket matrix coordinate pattern symmetric
2 2 2
1 1
2 1
`
	c2 := parse(t, pat)
	m2, err := c2.ToCSR()
	if err != nil {
		t.Fatal(err)
	}
	if m2.Values[m2.RowPtr[0]] != 1 {
		t.Fatalf("pattern entries should be 1.0, got %v", m2.Values)
	}
}

func TestParseMalformed(t *testing.T) {
	expectKind(t, "not a banner\n", sparse.KindFormat)
	expectKind(t, "%%MatrixMarket matrix coordinate complex general\n2 2 1\n1 1 1\n", sparse.KindFormat)
	expectKind(t, "%%MatrixWare matrix coordinate real general\n2 2 1\n1 1 1\n", sparse.KindFormat)
	expectKind(t, "%%MatrixMarket matrix coordinate real general\n2 2 x\n", sparse.KindFormat)
	expectKind(t, "%%MatrixMarket matrix coordinate real general\n2 2 2\n1 1 1\n", sparse.KindStructure)
	expectKind(t, "%%MatrixMarket matrix coordinate real general\n2 2 1\n3 1 1\n", sparse.KindStructure)
	expectKind(t, "%%MatrixMarket matrix coordinate real general\n2 2 2\n1 1 1\n1 1 2\n", sparse.KindStructure)
	expectKind(t, "%%MatrixMarket matrix coordinate real symmetric\n2 3 1\n1 2 1\n", sparse.KindStructure)
}

func TestReadFile(t *testing.T) {
	dir := t.TempDir()
	path := filepath.Join(dir, "a.mtx")
	content := "%%MatrixMarket matrix coordinate real general\n1 1 1\n1 1 2\n"
	if err := os.WriteFile(path, []byte(content), 0o644); err != nil {
		t.Fatal(err)
	}
	c, err := ReadFile(path)
	if err != nil {
		t.Fatal(err)
	}
	if c.Rows != 1 {
		t.Fatalf("rows = %d", c.Rows)
	}
	if _, err := ReadFile(filepath.Join(dir, "missing.mtx")); err == nil {
		t.Fatal("expected open error")
	}
}
