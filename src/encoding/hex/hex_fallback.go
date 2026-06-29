// Copyright 2026 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

//go:build !goexperiment.simd || (!arm64 && !amd64 && !wasm)

package hex

func encodeSIMD(dst, src []byte) int {
	return 0
}

func decodeSIMD(dst, src []byte) int {
	return 0
}
