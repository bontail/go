// Copyright 2026 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

//go:build goexperiment.simd && amd64

package hex

import "simd/archsimd"

const (
	simdBlockSize = 16
	avx2BlockSize = 32
)

var (
	hexTable128 = [simdBlockSize]byte{
		'0', '1', '2', '3', '4', '5', '6', '7',
		'8', '9', 'a', 'b', 'c', 'd', 'e', 'f',
	}
	// AVX2 byte permutations operate independently on each 128-bit lane.
	hexTable256 = [avx2BlockSize]byte{
		'0', '1', '2', '3', '4', '5', '6', '7',
		'8', '9', 'a', 'b', 'c', 'd', 'e', 'f',
		'0', '1', '2', '3', '4', '5', '6', '7',
		'8', '9', 'a', 'b', 'c', 'd', 'e', 'f',
	}
)

// Loading constants from memory keeps the 128-bit path compatible with AVX-only
// CPUs; the corresponding archsimd broadcast operations require AVX2.
var (
	lowNibbleMask128 = [simdBlockSize]byte{
		0x0f, 0x0f, 0x0f, 0x0f, 0x0f, 0x0f, 0x0f, 0x0f,
		0x0f, 0x0f, 0x0f, 0x0f, 0x0f, 0x0f, 0x0f, 0x0f,
	}
	lowNibbleMask256 = [avx2BlockSize]byte{
		0x0f, 0x0f, 0x0f, 0x0f, 0x0f, 0x0f, 0x0f, 0x0f,
		0x0f, 0x0f, 0x0f, 0x0f, 0x0f, 0x0f, 0x0f, 0x0f,
		0x0f, 0x0f, 0x0f, 0x0f, 0x0f, 0x0f, 0x0f, 0x0f,
		0x0f, 0x0f, 0x0f, 0x0f, 0x0f, 0x0f, 0x0f, 0x0f,
	}
	lowByteMask128 = [simdBlockSize / 2]uint16{
		0x00ff, 0x00ff, 0x00ff, 0x00ff, 0x00ff, 0x00ff, 0x00ff, 0x00ff,
	}
	lowByteMask256 = [avx2BlockSize / 2]uint16{
		0x00ff, 0x00ff, 0x00ff, 0x00ff, 0x00ff, 0x00ff, 0x00ff, 0x00ff,
		0x00ff, 0x00ff, 0x00ff, 0x00ff, 0x00ff, 0x00ff, 0x00ff, 0x00ff,
	}
	decodePackMask128 = [simdBlockSize]int8{
		0, 2, 4, 6, 8, 10, 12, 14,
		-1, -1, -1, -1, -1, -1, -1, -1,
	}
	decodePackMask256 = [avx2BlockSize]int8{
		0, 2, 4, 6, 8, 10, 12, 14,
		-1, -1, -1, -1, -1, -1, -1, -1,
		0, 2, 4, 6, 8, 10, 12, 14,
		-1, -1, -1, -1, -1, -1, -1, -1,
	}
	digitRangeStart128 = [simdBlockSize]int8{
		0x30, 0x30, 0x30, 0x30, 0x30, 0x30, 0x30, 0x30,
		0x30, 0x30, 0x30, 0x30, 0x30, 0x30, 0x30, 0x30,
	}
	digitRangeEnd128 = [simdBlockSize]int8{
		0x39, 0x39, 0x39, 0x39, 0x39, 0x39, 0x39, 0x39,
		0x39, 0x39, 0x39, 0x39, 0x39, 0x39, 0x39, 0x39,
	}
	upperLetterRangeStart128 = [simdBlockSize]int8{
		0x41, 0x41, 0x41, 0x41, 0x41, 0x41, 0x41, 0x41,
		0x41, 0x41, 0x41, 0x41, 0x41, 0x41, 0x41, 0x41,
	}
	upperLetterRangeEnd128 = [simdBlockSize]int8{
		0x46, 0x46, 0x46, 0x46, 0x46, 0x46, 0x46, 0x46,
		0x46, 0x46, 0x46, 0x46, 0x46, 0x46, 0x46, 0x46,
	}
	lowerLetterRangeStart128 = [simdBlockSize]int8{
		0x61, 0x61, 0x61, 0x61, 0x61, 0x61, 0x61, 0x61,
		0x61, 0x61, 0x61, 0x61, 0x61, 0x61, 0x61, 0x61,
	}
	lowerLetterRangeEnd128 = [simdBlockSize]int8{
		0x66, 0x66, 0x66, 0x66, 0x66, 0x66, 0x66, 0x66,
		0x66, 0x66, 0x66, 0x66, 0x66, 0x66, 0x66, 0x66,
	}
	// 0x61...0x66 - 0x57 = 0x0a...0x0f
	lowerLetterDeltaForSub128 = [simdBlockSize]int8{
		0x57, 0x57, 0x57, 0x57, 0x57, 0x57, 0x57, 0x57,
		0x57, 0x57, 0x57, 0x57, 0x57, 0x57, 0x57, 0x57,
	}
	normalizeLetterValue128 = [simdBlockSize]int8{
		0x20, 0x20, 0x20, 0x20, 0x20, 0x20, 0x20, 0x20,
		0x20, 0x20, 0x20, 0x20, 0x20, 0x20, 0x20, 0x20,
	}
)

// encodeSIMD encodes complete SIMD blocks and returns the number of source
// bytes consumed.
func encodeSIMD(dst, src []byte) int {
	if len(src) < simdBlockSize || !archsimd.X86.AVX() {
		return 0
	}

	processed := 0
	useAVX2 := archsimd.X86.AVX2() && len(src) >= avx2BlockSize
	if useAVX2 {
		for len(src) >= avx2BlockSize {
			encodedLo, encodedHi := encodeBlockAVX2(archsimd.LoadUint8x32(src))
			encodedLo.Store(dst[:avx2BlockSize])
			encodedHi.Store(dst[avx2BlockSize : 2*avx2BlockSize])

			src = src[avx2BlockSize:]
			dst = dst[2*avx2BlockSize:]
			processed += avx2BlockSize
		}
	}

	for len(src) >= simdBlockSize {
		encodedLo, encodedHi := encodeBlock(archsimd.LoadUint8x16(src))
		encodedLo.Store(dst[:simdBlockSize])
		encodedHi.Store(dst[simdBlockSize : 2*simdBlockSize])

		src = src[simdBlockSize:]
		dst = dst[2*simdBlockSize:]
		processed += simdBlockSize
	}

	if useAVX2 {
		archsimd.ClearAVXUpperBits()
	}
	return processed
}

// encodeBlock encodes one 128-bit vector into two consecutive vectors of
// hexadecimal digits using AVX instructions.
func encodeBlock(x archsimd.Uint8x16) (encodedLo, encodedHi archsimd.Uint8x16) {
	nibbleMask := archsimd.LoadUint8x16Array(&lowNibbleMask128)
	lowNibbles := x.And(nibbleMask)
	highNibbles := x.AsUint16x8().ShiftAllRight(4).AsUint8x16().And(nibbleMask)

	hexTable := archsimd.LoadUint8x16Array(&hexTable128)
	lowDigits := hexTable.PermuteOrZero(lowNibbles.AsInt8x16())
	highDigits := hexTable.PermuteOrZero(highNibbles.AsInt8x16())

	lowByteMask := archsimd.LoadUint16x8Array(&lowByteMask128)
	highWords := highDigits.AsUint16x8()
	lowWords := lowDigits.AsUint16x8()
	evenPairs := highWords.And(lowByteMask).Or(lowWords.ShiftAllLeft(8))
	oddPairs := highWords.ShiftAllRight(8).Or(lowWords.And(lowByteMask.ShiftAllLeft(8)))

	return evenPairs.InterleaveLo(oddPairs).AsUint8x16(), evenPairs.InterleaveHi(oddPairs).AsUint8x16()
}

// encodeBlockAVX2 encodes one 256-bit vector into two consecutive vectors of
// hexadecimal digits.
func encodeBlockAVX2(x archsimd.Uint8x32) (encodedLo, encodedHi archsimd.Uint8x32) {
	nibbleMask := archsimd.LoadUint8x32Array(&lowNibbleMask256)
	lowNibbles := x.And(nibbleMask)
	highNibbles := x.AsUint16x16().ShiftAllRight(4).AsUint8x32().And(nibbleMask)

	hexTable := archsimd.LoadUint8x32Array(&hexTable256)
	lowDigits := hexTable.PermuteOrZeroGrouped(lowNibbles.AsInt8x32())
	highDigits := hexTable.PermuteOrZeroGrouped(highNibbles.AsInt8x32())

	lowByteMask := archsimd.LoadUint16x16Array(&lowByteMask256)
	highWords := highDigits.AsUint16x16()
	lowWords := lowDigits.AsUint16x16()
	evenPairs := highWords.And(lowByteMask).Or(lowWords.ShiftAllLeft(8))
	oddPairs := highWords.ShiftAllRight(8).Or(lowWords.And(lowByteMask.ShiftAllLeft(8)))

	interleavedLo := evenPairs.InterleaveLoGrouped(oddPairs)
	interleavedHi := evenPairs.InterleaveHiGrouped(oddPairs)
	encodedLo = interleavedLo.ConcatPermute128Scalars(0, 2, interleavedHi).AsUint8x32()
	encodedHi = interleavedLo.ConcatPermute128Scalars(1, 3, interleavedHi).AsUint8x32()
	return encodedLo, encodedHi
}

// decodeSIMD decodes complete SIMD blocks and returns the number of source
// bytes consumed.
func decodeSIMD(dst, src []byte) (processed int) {
	if len(src) < simdBlockSize || !archsimd.X86.AVX() {
		return 0
	}

	useAVX2 := archsimd.X86.AVX2() && len(src) >= avx2BlockSize
	if useAVX2 {
		for len(src) >= avx2BlockSize {
			if invalid := decodeBlockAVX2(archsimd.LoadUint8x32(src), dst); invalid {
				archsimd.ClearAVXUpperBits()
				return processed
			}
			src = src[avx2BlockSize:]
			dst = dst[avx2BlockSize/2:]
			processed += avx2BlockSize
		}
	}

	for len(src) >= simdBlockSize {
		if invalid := decodeBlock(archsimd.LoadUint8x16(src), dst); invalid {
			if useAVX2 {
				archsimd.ClearAVXUpperBits()
			}
			return processed
		}
		src = src[simdBlockSize:]
		dst = dst[simdBlockSize/2:]
		processed += simdBlockSize
	}

	if useAVX2 {
		archsimd.ClearAVXUpperBits()
	}
	return processed
}

// decodeBlock decodes one 128-bit vector of hexadecimal digits into eight bytes.
// It returns whether the input contained an invalid byte.
func decodeBlock(input archsimd.Uint8x16, dst []byte) (invalid bool) {
	c := input.BitsToInt8()
	digitRangeStart := archsimd.LoadInt8x16Array(&digitRangeStart128)
	digitRangeEnd := archsimd.LoadInt8x16Array(&digitRangeEnd128)
	upperLetterRangeStart := archsimd.LoadInt8x16Array(&upperLetterRangeStart128)
	upperLetterRangeEnd := archsimd.LoadInt8x16Array(&upperLetterRangeEnd128)
	lowerLetterRangeStart := archsimd.LoadInt8x16Array(&lowerLetterRangeStart128)
	lowerLetterRangeEnd := archsimd.LoadInt8x16Array(&lowerLetterRangeEnd128)

	isDigit := c.GreaterEqual(digitRangeStart).
		And(digitRangeEnd.GreaterEqual(c))
	isUpper := c.GreaterEqual(upperLetterRangeStart).
		And(upperLetterRangeEnd.GreaterEqual(c))
	isLower := c.GreaterEqual(lowerLetterRangeStart).
		And(lowerLetterRangeEnd.GreaterEqual(c))
	isValid := isDigit.Or(isUpper.Or(isLower))

	if isValid.ToBits() != 0xffff {
		return true
	}

	lower := input.Or(archsimd.LoadInt8x16Array(&normalizeLetterValue128).AsUint8x16())
	letterNibble := lower.Sub(archsimd.LoadInt8x16Array(&lowerLetterDeltaForSub128).AsUint8x16())
	digitNibble := input.Sub(digitRangeStart.AsUint8x16())

	digitMask := isDigit.ToInt8x16()
	letterMask := digitMask.Not()
	nibble := digitNibble.BitsToInt8().And(digitMask).
		Or(letterNibble.BitsToInt8().And(letterMask))

	decodeNibbles(nibble.ToBits(), dst)
	return false
}

// decodeBlockAVX2 decodes one 256-bit vector of hexadecimal digits into
// sixteen bytes using AVX2 instructions.
// It returns whether the input contained an invalid byte.
func decodeBlockAVX2(input archsimd.Uint8x32, dst []byte) (invalid bool) {
	c := input.BitsToInt8()
	isDigit := c.GreaterEqual(archsimd.BroadcastInt8x32(0x30)).
		And(archsimd.BroadcastInt8x32(0x39).GreaterEqual(c))
	isUpper := c.GreaterEqual(archsimd.BroadcastInt8x32(0x41)).
		And(archsimd.BroadcastInt8x32(0x46).GreaterEqual(c))
	isLower := c.GreaterEqual(archsimd.BroadcastInt8x32(0x61)).
		And(archsimd.BroadcastInt8x32(0x66).GreaterEqual(c))
	isValid := isDigit.Or(isUpper.Or(isLower))

	if isValid.ToBits() != 0xffffffff {
		return true
	}

	lower := input.Or(archsimd.BroadcastUint8x32(0x20))
	letterNibble := lower.Sub(archsimd.BroadcastUint8x32(0x57))
	digitNibble := input.Sub(archsimd.BroadcastUint8x32(0x30))

	digitMask := isDigit.ToInt8x32()
	letterMask := digitMask.Not()
	nibble := digitNibble.BitsToInt8().And(digitMask).
		Or(letterNibble.BitsToInt8().And(letterMask)).ToBits()

	words := nibble.AsUint16x16()
	packed := words.And(archsimd.LoadUint16x16Array(&lowByteMask256)).ShiftAllLeft(4).
		Or(words.ShiftAllRight(8)).AsUint8x32().
		PermuteOrZeroGrouped(archsimd.LoadInt8x32Array(&decodePackMask256))
	storeUint64LE(dst, packed.GetLo().AsUint64x2().GetElem(0))
	storeUint64LE(dst[8:], packed.GetHi().AsUint64x2().GetElem(0))
	return false
}

// decodeNibbles combines 16 nibble values into 8 decoded bytes.
func decodeNibbles(nibble archsimd.Uint8x16, dst []byte) {
	words := nibble.AsUint16x8()
	packed := words.And(archsimd.LoadUint16x8Array(&lowByteMask128)).ShiftAllLeft(4).
		Or(words.ShiftAllRight(8)).AsUint8x16().
		PermuteOrZero(archsimd.LoadInt8x16Array(&decodePackMask128))
	storeUint64LE(dst, packed.AsUint64x2().GetElem(0))
}
