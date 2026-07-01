// Copyright 2026 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

//go:build goexperiment.simd && amd64

package hex

import (
	"internal/byteorder"
	"simd/archsimd"
)

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
	lowByteMask128 = [simdBlockSize / 2]uint16{
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
	decodePairWeights128 = [simdBlockSize]int8{
		16, 1, 16, 1, 16, 1, 16, 1,
		16, 1, 16, 1, 16, 1, 16, 1,
	}
	letterNibbleOffset128 = [simdBlockSize]byte{
		9, 9, 9, 9, 9, 9, 9, 9,
		9, 9, 9, 9, 9, 9, 9, 9,
	}
	digitRangeBefore128 = [simdBlockSize]int8{
		0x2f, 0x2f, 0x2f, 0x2f, 0x2f, 0x2f, 0x2f, 0x2f,
		0x2f, 0x2f, 0x2f, 0x2f, 0x2f, 0x2f, 0x2f, 0x2f,
	}
	digitRangeAfter128 = [simdBlockSize]int8{
		0x3a, 0x3a, 0x3a, 0x3a, 0x3a, 0x3a, 0x3a, 0x3a,
		0x3a, 0x3a, 0x3a, 0x3a, 0x3a, 0x3a, 0x3a, 0x3a,
	}
	lowerLetterRangeBefore128 = [simdBlockSize]int8{
		0x60, 0x60, 0x60, 0x60, 0x60, 0x60, 0x60, 0x60,
		0x60, 0x60, 0x60, 0x60, 0x60, 0x60, 0x60, 0x60,
	}
	lowerLetterRangeAfter128 = [simdBlockSize]int8{
		0x67, 0x67, 0x67, 0x67, 0x67, 0x67, 0x67, 0x67,
		0x67, 0x67, 0x67, 0x67, 0x67, 0x67, 0x67, 0x67,
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
		nibbleMask := archsimd.BroadcastUint8x32(0x0f)
		hexTable := archsimd.LoadUint8x32Array(&hexTable256)
		lowByteMask := archsimd.BroadcastUint16x16(0x00ff)
		for len(src) >= avx2BlockSize {
			x := archsimd.LoadUint8x32(src)
			lowNibbles := x.And(nibbleMask)
			highNibbles := x.AsUint16x16().ShiftAllRight(4).AsUint8x32().And(nibbleMask)
			lowDigits := hexTable.PermuteOrZeroGrouped(lowNibbles.AsInt8x32())
			highDigits := hexTable.PermuteOrZeroGrouped(highNibbles.AsInt8x32())
			highWords := highDigits.AsUint16x16()
			lowWords := lowDigits.AsUint16x16()
			evenPairs := highWords.And(lowByteMask).Or(lowWords.ShiftAllLeft(8))
			oddPairs := highWords.ShiftAllRight(8).Or(lowWords.And(lowByteMask.ShiftAllLeft(8)))
			interleavedLo := evenPairs.InterleaveLoGrouped(oddPairs)
			interleavedHi := evenPairs.InterleaveHiGrouped(oddPairs)
			encodedLo := interleavedLo.ConcatPermute128Scalars(0, 2, interleavedHi).AsUint8x32()
			encodedHi := interleavedLo.ConcatPermute128Scalars(1, 3, interleavedHi).AsUint8x32()
			encodedLo.Store(dst[:avx2BlockSize])
			encodedHi.Store(dst[avx2BlockSize : 2*avx2BlockSize])

			src = src[avx2BlockSize:]
			dst = dst[2*avx2BlockSize:]
			processed += avx2BlockSize
		}
	}

	nibbleMask := archsimd.LoadUint8x16Array(&lowNibbleMask128)
	hexTable := archsimd.LoadUint8x16Array(&hexTable128)
	lowByteMask := archsimd.LoadUint16x8Array(&lowByteMask128)
	for len(src) >= simdBlockSize {
		x := archsimd.LoadUint8x16(src)
		lowNibbles := x.And(nibbleMask)
		highNibbles := x.AsUint16x8().ShiftAllRight(4).AsUint8x16().And(nibbleMask)
		lowDigits := hexTable.PermuteOrZero(lowNibbles.AsInt8x16())
		highDigits := hexTable.PermuteOrZero(highNibbles.AsInt8x16())
		highWords := highDigits.AsUint16x8()
		lowWords := lowDigits.AsUint16x8()
		evenPairs := highWords.And(lowByteMask).Or(lowWords.ShiftAllLeft(8))
		oddPairs := highWords.ShiftAllRight(8).Or(lowWords.And(lowByteMask.ShiftAllLeft(8)))
		encodedLo := evenPairs.InterleaveLo(oddPairs).AsUint8x16()
		encodedHi := evenPairs.InterleaveHi(oddPairs).AsUint8x16()
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

// decodeSIMD decodes complete SIMD blocks and returns the number of source
// bytes consumed.
func decodeSIMD(dst, src []byte) (processed int) {
	if len(src) < simdBlockSize || !archsimd.X86.AVX() {
		return 0
	}

	useAVX2 := archsimd.X86.AVX2() && len(src) >= avx2BlockSize
	if useAVX2 {
		normalizeLetterValue := archsimd.BroadcastUint8x32(0x20)
		digitRangeBefore := archsimd.BroadcastInt8x32(0x2f)
		digitRangeAfter := archsimd.BroadcastInt8x32(0x3a)
		lowerLetterRangeBefore := archsimd.BroadcastInt8x32(0x60)
		lowerLetterRangeAfter := archsimd.BroadcastInt8x32(0x67)
		lowNibbleMask := archsimd.BroadcastUint8x32(0x0f)
		letterNibbleOffset := archsimd.BroadcastUint8x32(9)
		decodePairWeights := archsimd.BroadcastUint16x16(0x0110).AsInt8x32()
		decodePackMask := archsimd.LoadInt8x32Array(&decodePackMask256)
		for len(src) >= avx2BlockSize {
			input := archsimd.LoadUint8x32(src)
			c := input.BitsToInt8()
			lowerC := input.Or(normalizeLetterValue).BitsToInt8()
			isDigit := c.Greater(digitRangeBefore).
				And(digitRangeAfter.Greater(c))
			isLetter := lowerC.Greater(lowerLetterRangeBefore).
				And(lowerLetterRangeAfter.Greater(lowerC))
			isValid := isDigit.Or(isLetter)

			if isValid.ToBits() != 0xffffffff {
				archsimd.ClearAVXUpperBits()
				return processed
			}

			nibble := input.And(lowNibbleMask).
				Add(isLetter.ToInt8x32().ToBits().And(letterNibbleOffset))
			packed := nibble.DotProductPairsSaturated(decodePairWeights).
				ToBits().AsUint8x32().
				PermuteOrZeroGrouped(decodePackMask)
			decoded := packed.GetLo().AsUint64x2().
				ConcatPermuteScalars(0, 2, packed.GetHi().AsUint64x2()).AsUint8x16()
			decoded.Store(dst[:16])
			src = src[avx2BlockSize:]
			dst = dst[avx2BlockSize/2:]
			processed += avx2BlockSize
		}
	}

	digitRangeBefore := archsimd.LoadInt8x16Array(&digitRangeBefore128)
	digitRangeAfter := archsimd.LoadInt8x16Array(&digitRangeAfter128)
	lowerLetterRangeBefore := archsimd.LoadInt8x16Array(&lowerLetterRangeBefore128)
	lowerLetterRangeAfter := archsimd.LoadInt8x16Array(&lowerLetterRangeAfter128)
	normalizeLetterValue := archsimd.LoadInt8x16Array(&normalizeLetterValue128).AsUint8x16()
	lowNibbleMask := archsimd.LoadUint8x16Array(&lowNibbleMask128)
	letterNibbleOffset := archsimd.LoadUint8x16Array(&letterNibbleOffset128)
	decodePairWeights := archsimd.LoadInt8x16Array(&decodePairWeights128)
	decodePackMask := archsimd.LoadInt8x16Array(&decodePackMask128)
	for len(src) >= simdBlockSize {
		input := archsimd.LoadUint8x16(src)
		c := input.BitsToInt8()
		lowerC := input.Or(normalizeLetterValue).BitsToInt8()
		isDigit := c.Greater(digitRangeBefore).
			And(digitRangeAfter.Greater(c))
		isLetter := lowerC.Greater(lowerLetterRangeBefore).
			And(lowerLetterRangeAfter.Greater(lowerC))
		isValid := isDigit.Or(isLetter)

		if isValid.ToBits() != 0xffff {
			if useAVX2 {
				archsimd.ClearAVXUpperBits()
			}
			return processed
		}

		nibble := input.And(lowNibbleMask).
			Add(isLetter.ToInt8x16().ToBits().And(letterNibbleOffset))
		packed := nibble.DotProductPairsSaturated(decodePairWeights).
			ToBits().AsUint8x16().
			PermuteOrZero(decodePackMask)
		byteorder.LEPutUint64(dst, packed.AsUint64x2().GetElem(0))
		src = src[simdBlockSize:]
		dst = dst[simdBlockSize/2:]
		processed += simdBlockSize
	}

	if useAVX2 {
		archsimd.ClearAVXUpperBits()
	}
	return processed
}
