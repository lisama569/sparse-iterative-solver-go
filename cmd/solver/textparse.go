package main

import (
	"strconv"
	"strings"
)

func trimSpace(s string) string { return strings.TrimSpace(s) }

func indexByte(s string, b byte) int { return strings.IndexByte(s, b) }

func parseFloat(s string, out *float64) (int, error) {
	v, err := strconv.ParseFloat(s, 64)
	if err != nil {
		return 0, err
	}
	*out = v
	return len(s), nil
}
