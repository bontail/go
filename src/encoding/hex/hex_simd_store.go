// Copyright 2026 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

//go:build goexperiment.simd && (amd64 || wasm)

package hex

// storeUint64LE stores value into dst in little-endian byte order.
func storeUint64LE(dst []byte, value uint64) {
	_ = dst[7]
	dst[0] = byte(value)
	dst[1] = byte(value >> 8)
	dst[2] = byte(value >> 16)
	dst[3] = byte(value >> 24)
	dst[4] = byte(value >> 32)
	dst[5] = byte(value >> 40)
	dst[6] = byte(value >> 48)
	dst[7] = byte(value >> 56)
}
