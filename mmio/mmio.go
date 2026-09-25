// Package mmio reads Matrix Market (.mtx) coordinate matrices from local
// files or byte slices. Only the subset needed for real-valued sparse
// linear systems is supported:
//
//	%%MatrixMarket matrix coordinate {real|integer|pattern} {general|symmetric}
//
// Indices in the file are 1-based as required by the format and converted
// to zero-based triplets. Anything the parser cannot interpret is reported
// as a sparse.KindFormat error; conflicts discovered while validating the
// assembled matrix (bad dimensions, duplicates, out-of-range indices,
// non-finite values) surface as KindStructure / KindNumeric instead.
package mmio

import (
	"bufio"
	"fmt"
	"io"
	"os"
	"strconv"
	"strings"

	"github.com/lisama569/sparse-iterative-solver-go/sparse"
)

// ReadCOO parses a Matrix Market coordinate matrix from r.
func ReadCOO(r io.Reader) (*sparse.COO, error) {
	br := bufio.NewReader(r)
	var headerOK bool
	var field, symmetry string
	var lineNo int

	// Banner must be the first non-empty line (comments may precede it,
	// but the standard puts it first; we accept leading blank lines only).
	for {
		line, err := br.ReadString('\n')
		lineNo++
		trimmed := strings.TrimSpace(line)
		if trimmed == "" {
			if err == io.EOF {
				return nil, &sparse.InputError{Kind: sparse.KindFormat, Message: "empty input, missing Matrix Market banner"}
			}
			if err != nil {
				return nil, fmt.Errorf("reading banner: %w", err)
			}
			continue
		}
		if !strings.HasPrefix(trimmed, "%%MatrixMarket") {
			return nil, &sparse.InputError{Kind: sparse.KindFormat,
				Message: fmt.Sprintf("line %d is not a Matrix Market banner", lineNo)}
		}
		toks := strings.Fields(trimmed)
		// %%MatrixMarket matrix coordinate <field> <symmetry>
		if len(toks) < 5 || strings.ToLower(toks[1]) != "matrix" || strings.ToLower(toks[2]) != "coordinate" {
			return nil, &sparse.InputError{Kind: sparse.KindFormat,
				Message: fmt.Sprintf("line %d has unsupported banner %q", lineNo, trimmed)}
		}
		field = strings.ToLower(toks[3])
		symmetry = strings.ToLower(toks[4])
		switch field {
		case "real", "integer", "pattern":
		default:
			return nil, &sparse.InputError{Kind: sparse.KindFormat,
				Message: fmt.Sprintf("unsupported field type %q (want real, integer or pattern)", field)}
		}
		switch symmetry {
		case "general", "symmetric":
		default:
			return nil, &sparse.InputError{Kind: sparse.KindFormat,
				Message: fmt.Sprintf("unsupported symmetry %q (want general or symmetric)", symmetry)}
		}
		headerOK = true
		break
	}
	if !headerOK {
		return nil, fmt.Errorf("format error: missing Matrix Market banner")
	}

	// Skip comment / blank lines until the size line.
	var sizeLine string
	for {
		line, err := br.ReadString('\n')
		lineNo++
		trimmed := strings.TrimSpace(stripComment(line))
		if trimmed == "" {
			if err == io.EOF {
				return nil, &sparse.InputError{Kind: sparse.KindFormat, Message: "reached end of file before size line"}
			}
			if err != nil {
				return nil, fmt.Errorf("reading size line: %w", err)
			}
			continue
		}
		sizeLine = trimmed
		break
	}
	sizeToks := strings.Fields(sizeLine)
	if len(sizeToks) != 3 {
		return nil, &sparse.InputError{Kind: sparse.KindFormat,
			Message: fmt.Sprintf("size line %q must contain exactly 3 numbers", sizeLine)}
	}
	rows, err1 := strconv.Atoi(sizeToks[0])
	cols, err2 := strconv.Atoi(sizeToks[1])
	nnz, err3 := strconv.Atoi(sizeToks[2])
	if err1 != nil || err2 != nil || err3 != nil {
		return nil, &sparse.InputError{Kind: sparse.KindFormat,
			Message: fmt.Sprintf("size line %q has non-integer fields", sizeLine)}
	}
	if rows <= 0 || cols <= 0 || nnz < 0 {
		return nil, &sparse.InputError{Kind: sparse.KindStructure,
			Message: fmt.Sprintf("invalid declared size %d %d %d", rows, cols, nnz)}
	}

	wantTokens := 3
	if field == "pattern" {
		wantTokens = 2
	}
	triplets := make([]sparse.Triplet, 0, nnz)
	for len(triplets) < nnz {
		line, err := br.ReadString('\n')
		lineNo++
		if err != nil && err != io.EOF {
			return nil, fmt.Errorf("reading data line %d: %w", lineNo, err)
		}
		trimmed := strings.TrimSpace(stripComment(line))
		if trimmed == "" {
			if err == io.EOF {
				break
			}
			continue
		}
		toks := strings.Fields(trimmed)
		if len(toks) != wantTokens {
			return nil, &sparse.InputError{Kind: sparse.KindFormat,
				Message: fmt.Sprintf("line %d has %d fields, want %d", lineNo, len(toks), wantTokens)}
		}
		i, errI := strconv.Atoi(toks[0])
		j, errJ := strconv.Atoi(toks[1])
		if errI != nil || errJ != nil {
			return nil, &sparse.InputError{Kind: sparse.KindFormat,
				Message: fmt.Sprintf("line %d has non-integer indices %q %q", lineNo, toks[0], toks[1])}
		}
		var v float64
		if field == "pattern" {
			v = 1
		} else {
			v, err = strconv.ParseFloat(toks[2], 64)
			if err != nil {
				return nil, &sparse.InputError{Kind: sparse.KindFormat,
					Message: fmt.Sprintf("line %d has invalid value %q", lineNo, toks[2])}
			}
		}
		triplets = append(triplets, sparse.Triplet{Row: i - 1, Col: j - 1, Value: v})
		if err == io.EOF {
			break
		}
	}
	if len(triplets) != nnz {
		return nil, &sparse.InputError{Kind: sparse.KindStructure,
			Message: fmt.Sprintf("declared %d entries but found %d", nnz, len(triplets))}
	}

	coo := &sparse.COO{Rows: rows, Cols: cols, Triplets: triplets, Symmetric: symmetry == "symmetric"}
	// Validate fully at the parse boundary so callers never receive a COO
	// with out-of-range indices, duplicates or non-finite values.
	if _, err := coo.ToCSR(); err != nil {
		return nil, err
	}
	return coo, nil
}

// ReadFile parses a Matrix Market matrix from a local file path.
func ReadFile(path string) (*sparse.COO, error) {
	f, err := os.Open(path)
	if err != nil {
		return nil, fmt.Errorf("cannot open %q: %w", path, err)
	}
	defer f.Close()
	return ReadCOO(f)
}

// stripComment removes an in-line Matrix Market comment starting with '%'.
func stripComment(line string) string {
	if i := strings.IndexByte(line, '%'); i >= 0 {
		return line[:i]
	}
	return line
}
