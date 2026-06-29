// Copyright 2026 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

//go:build goexperiment.simd && (arm64 || amd64 || wasm)

package hex

import (
	"bytes"
	"strconv"
	"testing"
)

func TestEncodeSIMDBoundaries(t *testing.T) {
	src := make([]byte, 257)
	for i := range src {
		src[i] = byte(i)
	}

	sizes := []int{
		0, 1,
		15, 16, 17,
		31, 32, 33,
		47, 48, 49,
		63, 64, 65,
		127, 128, 129,
		255, 256, 257,
	}

	const guardSize = 16
	wantGuard := bytes.Repeat([]byte{0xa5}, guardSize)
	for _, size := range sizes {
		t.Run(strconv.Itoa(size), func(t *testing.T) {
			encodedLen := EncodedLen(size)
			backing := bytes.Repeat([]byte{0xa5}, encodedLen+2*guardSize)
			dst := backing[guardSize : guardSize+encodedLen]

			if got := Encode(dst, src[:size]); got != encodedLen {
				t.Fatalf("Encode returned %d, want %d", got, encodedLen)
			}
			if want := encodeScalarForTest(src[:size]); !bytes.Equal(dst, want) {
				t.Fatalf("Encode returned %q, want %q", dst, want)
			}
			if !bytes.Equal(backing[:guardSize], wantGuard) {
				t.Fatal("Encode wrote before dst")
			}
			if !bytes.Equal(backing[guardSize+encodedLen:], wantGuard) {
				t.Fatal("Encode wrote past dst")
			}
		})
	}
}

func TestDecodeSIMDBoundaries(t *testing.T) {
	sizes := []int{
		0, 2,
		14, 16, 18,
		30, 32, 34,
		46, 48, 50,
		62, 64, 66,
		126, 128, 130,
		254, 256, 258,
	}

	const guardSize = 16
	wantGuard := bytes.Repeat([]byte{0xa5}, guardSize)
	for _, size := range sizes {
		t.Run(strconv.Itoa(size), func(t *testing.T) {
			const hexChars = "0123456789abcdefABCDEF"
			src := make([]byte, size)
			for i := range src {
				src[i] = hexChars[i%len(hexChars)]
			}
			decodedLen := DecodedLen(size)

			backing := bytes.Repeat([]byte{0xa5}, decodedLen+2*guardSize)
			dst := backing[guardSize : guardSize+decodedLen]

			n, err := Decode(dst, src)
			if err != nil {
				t.Fatalf("Decode returned error: %v", err)
			}
			if n != decodedLen {
				t.Fatalf("Decode returned %d, want %d", n, decodedLen)
			}
			if want := decodeScalarForTest(src); !bytes.Equal(dst, want) {
				t.Fatalf("Decode returned %q, want %q", dst, want)
			}
			if !bytes.Equal(backing[:guardSize], wantGuard) {
				t.Fatal("Decode wrote before dst")
			}
			if !bytes.Equal(backing[guardSize+decodedLen:], wantGuard) {
				t.Fatal("Decode wrote past dst")
			}
		})
	}
}

func TestDecodeSIMDOddLength(t *testing.T) {
	for _, size := range []int{1, 15, 17, 31, 33, 47, 49} {
		t.Run(strconv.Itoa(size), func(t *testing.T) {
			src := bytes.Repeat([]byte{'0'}, size)
			dst := bytes.Repeat([]byte{0xa5}, DecodedLen(size))

			n, err := Decode(dst, src)
			if err != ErrLength {
				t.Fatalf("Decode returned error %v, want ErrLength", err)
			}
			if n != DecodedLen(size) {
				t.Fatalf("Decode returned %d, want %d", n, DecodedLen(size))
			}
			if want := make([]byte, n); !bytes.Equal(dst, want) {
				t.Fatalf("Decode returned %q, want %q", dst, want)
			}
		})
	}
}

func TestDecodeSIMDErrors(t *testing.T) {
	tests := []struct {
		size int
		pos  int
	}{
		{32, 0},
		{32, 1},
		{32, 2},
		{32, 14},
		{32, 15},
		{32, 16},
		{32, 17},
		{32, 18},
		{32, 30},
		{32, 31},
		{33, 32},
	}
	for _, test := range tests {
		name := strconv.Itoa(test.pos) + "/" + strconv.Itoa(test.size)
		t.Run(name, func(t *testing.T) {
			src := bytes.Repeat([]byte{'0'}, test.size)
			src[test.pos] = 'z'
			dst := bytes.Repeat([]byte{0xa5}, DecodedLen(test.size))

			n, err := Decode(dst, src)
			if err != InvalidByteError('z') {
				t.Fatalf("Decode returned error %v, want InvalidByteError(%q)", err, 'z')
			}
			wantN := test.pos / 2
			if n != wantN {
				t.Fatalf("Decode returned %d, want %d", n, wantN)
			}
			if want := make([]byte, n); !bytes.Equal(dst[:n], want) {
				t.Fatalf("Decode returned %q, want %q", dst[:n], want)
			}
			if want := bytes.Repeat([]byte{0xa5}, len(dst)-n); !bytes.Equal(dst[n:], want) {
				t.Fatalf("Decode wrote past the valid prefix: got %x, want %x", dst[n:], want)
			}
		})
	}
}

func encodeScalarForTest(src []byte) []byte {
	dst := make([]byte, EncodedLen(len(src)))
	for i, b := range src {
		dst[2*i] = hextable[b>>4]
		dst[2*i+1] = hextable[b&0x0f]
	}
	return dst
}

func decodeScalarForTest(src []byte) []byte {
	dst := make([]byte, DecodedLen(len(src)))
	for i := 0; i < len(src)-1; i += 2 {
		dst[i/2] = (reverseHexTable[src[i]] << 4) | reverseHexTable[src[i+1]]
	}
	return dst
}
