package sparse

import (
	"bufio"
	"fmt"
	"io"
	"strconv"
	"strings"
)

// LoadMatrixMarket 解析 Matrix Market 坐标格式。
// 支持 real/integer 数值、general/symmetric 结构；其余（complex/pattern、
// hermitian/skew）返回 ErrFormat。symmetric 输入按镜像展开为完整矩阵。
func LoadMatrixMarket(r io.Reader) (*Matrix, error) {
	sc := bufio.NewScanner(r)
	sc.Buffer(make([]byte, 64*1024), 1024*1024)

	// 头部：%%MatrixMarket matrix coordinate <field> <symmetry>
	if !sc.Scan() {
		return nil, fmt.Errorf("%w: 空输入，缺少 MatrixMarket 头部", ErrFormat)
	}
	header := strings.Fields(sc.Text())
	if len(header) != 5 || !strings.EqualFold(header[0], "%%MatrixMarket") ||
		!strings.EqualFold(header[1], "matrix") || !strings.EqualFold(header[2], "coordinate") {
		return nil, fmt.Errorf("%w: 非法头部 %q", ErrFormat, sc.Text())
	}
	field := strings.ToLower(header[3])
	if field != "real" && field != "integer" {
		return nil, fmt.Errorf("%w: 不支持的数值类型 %q（仅支持 real/integer）", ErrFormat, header[3])
	}
	sym := strings.ToLower(header[4])
	if sym != "general" && sym != "symmetric" {
		return nil, fmt.Errorf("%w: 不支持的对称性 %q（仅支持 general/symmetric）", ErrFormat, header[4])
	}

	// 尺寸行：跳过注释与空行。
	var rows, cols, nnz int
	for sc.Scan() {
		line := strings.TrimSpace(sc.Text())
		if line == "" || strings.HasPrefix(line, "%") {
			continue
		}
		f := strings.Fields(line)
		if len(f) != 3 {
			return nil, fmt.Errorf("%w: 尺寸行应有 3 个字段, got %q", ErrFormat, line)
		}
		var err error
		if rows, err = strconv.Atoi(f[0]); err != nil {
			return nil, fmt.Errorf("%w: 行数解析失败: %v", ErrFormat, err)
		}
		if cols, err = strconv.Atoi(f[1]); err != nil {
			return nil, fmt.Errorf("%w: 列数解析失败: %v", ErrFormat, err)
		}
		if nnz, err = strconv.Atoi(f[2]); err != nil {
			return nil, fmt.Errorf("%w: 非零元个数解析失败: %v", ErrFormat, err)
		}
		if rows <= 0 || cols <= 0 || nnz < 0 {
			return nil, fmt.Errorf("%w: 尺寸必须为正且 nnz 非负, got %d %d %d",
				ErrDimension, rows, cols, nnz)
		}
		break
	}

	entries := make([]Entry, 0, nnz)
	count := 0
	lineNo := 0
	for sc.Scan() {
		lineNo++
		line := strings.TrimSpace(sc.Text())
		if line == "" || strings.HasPrefix(line, "%") {
			continue
		}
		f := strings.Fields(line)
		if len(f) != 3 {
			return nil, fmt.Errorf("%w: 数据行 %d 应有 3 个字段, got %q", ErrFormat, lineNo, line)
		}
		i, err1 := strconv.Atoi(f[0])
		j, err2 := strconv.Atoi(f[1])
		v, err3 := strconv.ParseFloat(f[2], 64)
		if err1 != nil || err2 != nil || err3 != nil {
			return nil, fmt.Errorf("%w: 数据行 %d 解析失败: %q", ErrFormat, lineNo, line)
		}
		// Matrix Market 是 1-based。
		if i < 1 || i > rows || j < 1 || j > cols {
			return nil, fmt.Errorf("%w: 数据行 %d 索引 (%d,%d) 超出 %d x %d",
				ErrBadIndex, lineNo, i, j, rows, cols)
		}
		entries = append(entries, Entry{Row: i - 1, Col: j - 1, Val: v})
		if sym == "symmetric" && i != j {
			entries = append(entries, Entry{Row: j - 1, Col: i - 1, Val: v})
		}
		count++
	}
	if err := sc.Err(); err != nil {
		return nil, fmt.Errorf("%w: 读取失败: %v", ErrFormat, err)
	}
	if count != nnz {
		return nil, fmt.Errorf("%w: 声明 %d 个非零元, 实际读到 %d 个", ErrFormat, nnz, count)
	}
	return BuildFromCOO(rows, cols, entries)
}

// LoadCOO 解析纯文本三元组：首行 "rows cols nnz"，其后每行 "i j v"（1-based）。
// 支持 # 开头的注释行。
func LoadCOO(r io.Reader) (*Matrix, error) {
	sc := bufio.NewScanner(r)
	sc.Buffer(make([]byte, 64*1024), 1024*1024)

	next := func() (string, bool) {
		for sc.Scan() {
			line := strings.TrimSpace(sc.Text())
			if line == "" || strings.HasPrefix(line, "#") {
				continue
			}
			return line, true
		}
		return "", false
	}

	line, ok := next()
	if !ok {
		return nil, fmt.Errorf("%w: 空输入，缺少尺寸行", ErrFormat)
	}
	f := strings.Fields(line)
	if len(f) != 3 {
		return nil, fmt.Errorf("%w: 尺寸行应有 3 个字段, got %q", ErrFormat, line)
	}
	rows, err1 := strconv.Atoi(f[0])
	cols, err2 := strconv.Atoi(f[1])
	nnz, err3 := strconv.Atoi(f[2])
	if err1 != nil || err2 != nil || err3 != nil {
		return nil, fmt.Errorf("%w: 尺寸行解析失败: %q", ErrFormat, line)
	}
	if rows <= 0 || cols <= 0 || nnz < 0 {
		return nil, fmt.Errorf("%w: 尺寸必须为正且 nnz 非负, got %d %d %d",
			ErrDimension, rows, cols, nnz)
	}

	entries := make([]Entry, 0, nnz)
	count := 0
	for {
		line, ok := next()
		if !ok {
			break
		}
		f := strings.Fields(line)
		if len(f) != 3 {
			return nil, fmt.Errorf("%w: 数据行应有 3 个字段, got %q", ErrFormat, line)
		}
		i, err1 := strconv.Atoi(f[0])
		j, err2 := strconv.Atoi(f[1])
		v, err3 := strconv.ParseFloat(f[2], 64)
		if err1 != nil || err2 != nil || err3 != nil {
			return nil, fmt.Errorf("%w: 数据行解析失败: %q", ErrFormat, line)
		}
		if i < 1 || i > rows || j < 1 || j > cols {
			return nil, fmt.Errorf("%w: 索引 (%d,%d) 超出 %d x %d", ErrBadIndex, i, j, rows, cols)
		}
		entries = append(entries, Entry{Row: i - 1, Col: j - 1, Val: v})
		count++
	}
	if err := sc.Err(); err != nil {
		return nil, fmt.Errorf("%w: 读取失败: %v", ErrFormat, err)
	}
	if count != nnz {
		return nil, fmt.Errorf("%w: 声明 %d 个非零元, 实际读到 %d 个", ErrFormat, nnz, count)
	}
	return BuildFromCOO(rows, cols, entries)
}
