package main

import (
	"os"
	"os/exec"
	"path/filepath"
	"strings"
	"testing"
)

// buildCLI builds the solver binary once per test run into the temp dir.
func buildCLI(t *testing.T) string {
	t.Helper()
	bin := filepath.Join(t.TempDir(), "solver")
	cmd := exec.Command("go", "build", "-o", bin, ".")
	if out, err := cmd.CombinedOutput(); err != nil {
		t.Fatalf("build failed: %v\n%s", err, out)
	}
	return bin
}

func writeFile(t *testing.T, dir, name, content string) string {
	t.Helper()
	p := filepath.Join(dir, name)
	if err := os.WriteFile(p, []byte(content), 0o644); err != nil {
		t.Fatal(err)
	}
	return p
}

const spdMTX = `%%MatrixMarket matrix coordinate real symmetric
4 4 7
1 1 4
2 1 -1
2 2 4
3 2 -1
3 3 4
4 3 -1
4 4 4
`

const badMTX = "not a matrix market file\n"

func TestCLIConverged(t *testing.T) {
	bin := buildCLI(t)
	dir := t.TempDir()
	mp := writeFile(t, dir, "A.mtx", spdMTX)
	rp := writeFile(t, dir, "b.txt", "1\n2\n3\n4\n")
	out, err := exec.Command(bin, "-matrix", mp, "-rhs", rp, "-tol", "1e-9").CombinedOutput()
	if err != nil {
		t.Fatalf("expected exit 0, got %v: %s", err, out)
	}
	if !strings.Contains(string(out), "status=converged") {
		t.Fatalf("missing converged marker: %s", out)
	}
}

func TestCLIMaxIterations(t *testing.T) {
	bin := buildCLI(t)
	dir := t.TempDir()
	mp := writeFile(t, dir, "A.mtx", spdMTX)
	rp := writeFile(t, dir, "b.txt", "1\n2\n3\n4\n")
	cmd := exec.Command(bin, "-matrix", mp, "-rhs", rp, "-tol", "1e-16", "-maxiter", "1")
	out, err := cmd.CombinedOutput()
	if !strings.Contains(string(out), "status=max_iterations") {
		t.Fatalf("missing max_iterations marker: %s", out)
	}
	if code := exitCode(t, err); code != 1 {
		t.Fatalf("exit code = %d, want 1", code)
	}
}

func TestCLIFormatErrorExit2(t *testing.T) {
	bin := buildCLI(t)
	dir := t.TempDir()
	mp := writeFile(t, dir, "bad.mtx", badMTX)
	cmd := exec.Command(bin, "-matrix", mp)
	out, err := cmd.CombinedOutput()
	if err == nil {
		t.Fatalf("expected non-zero exit, output: %s", out)
	}
	if !strings.Contains(string(out), "[format]") {
		t.Fatalf("error not classified as format: %s", out)
	}
	if code := exitCode(t, err); code != 2 {
		t.Fatalf("exit code = %d, want 2", code)
	}
}

func exitCode(t *testing.T, err error) int {
	t.Helper()
	if ee, ok := err.(*exec.ExitError); ok {
		return ee.ExitCode()
	}
	return -1
}
