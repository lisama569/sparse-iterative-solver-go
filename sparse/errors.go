package sparse

import "fmt"

// ErrorKind classifies errors produced while parsing or validating sparse
// inputs so callers can distinguish malformed format, structural conflicts
// and unusable numeric values.
type ErrorKind int

const (
	// KindFormat describes text that cannot be parsed as the declared format.
	KindFormat ErrorKind = iota + 1
	// KindStructure describes a structurally inconsistent matrix
	// (bad dimensions, out-of-range indices, duplicate entries, ...).
	KindStructure
	// KindNumeric describes a matrix whose values make solving impossible
	// (NaN/Inf, zero/non-positive diagonal for a symmetric system, ...).
	KindNumeric
)

// InputError carries a machine-readable category alongside the message.
type InputError struct {
	Kind    ErrorKind
	Message string
}

func (e *InputError) Error() string { return kindName(e.Kind) + ": " + e.Message }

// String renders the machine-readable category for reports and logs.
func (k ErrorKind) String() string { return kindName(k) }

func kindName(k ErrorKind) string {
	switch k {
	case KindFormat:
		return "format error"
	case KindStructure:
		return "structure error"
	case KindNumeric:
		return "numeric error"
	default:
		return "error"
	}
}

func newError(kind ErrorKind, format string, args ...any) error {
	return &InputError{Kind: kind, Message: fmt.Sprintf(format, args...)}
}
