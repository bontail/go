// Copyright 2026 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

//go:build goexperiment.simd && amd64

package hex

import (
	"internal/byteorder"
	"simd/archsimd"
)

const simdBlockSize = 16
const simdBlockSizeAVX2 = 32

const lowNibbleMask = 0x0f
const lowLetter = 0x20 // 'A'...'F' -> 'a'...'f'.
const digitRangeStart = byte('0')
const digitRangeSize = byte('9') - digitRangeStart + 1
const letterRangeStart = byte('a')
const letterRangeSize = byte('f') - letterRangeStart + 1
const letterNibbleOffset = 9

var (
	hexTableData = [simdBlockSize]byte{
		'0', '1', '2', '3', '4', '5', '6', '7',
		'8', '9', 'a', 'b', 'c', 'd', 'e', 'f',
	}
	hexTableDataAVX2 = [simdBlockSizeAVX2]byte{
		'0', '1', '2', '3', '4', '5', '6', '7',
		'8', '9', 'a', 'b', 'c', 'd', 'e', 'f',
		'0', '1', '2', '3', '4', '5', '6', '7',
		'8', '9', 'a', 'b', 'c', 'd', 'e', 'f',
	}

	// The 128-bit path must remain compatible with AVX-only CPUs. In
	// particular, archsimd's 128-bit broadcast helpers require AVX2, so load
	// these constants from memory instead.
	lowNibbleMaskData = [simdBlockSize]byte{
		0x0f, 0x0f, 0x0f, 0x0f, 0x0f, 0x0f, 0x0f, 0x0f,
		0x0f, 0x0f, 0x0f, 0x0f, 0x0f, 0x0f, 0x0f, 0x0f,
	}
	normalizeLetterValueData = [simdBlockSize]byte{
		0x20, 0x20, 0x20, 0x20, 0x20, 0x20, 0x20, 0x20,
		0x20, 0x20, 0x20, 0x20, 0x20, 0x20, 0x20, 0x20,
	}
	digitRangeBeforeData = [simdBlockSize]int8{
		0x2f, 0x2f, 0x2f, 0x2f, 0x2f, 0x2f, 0x2f, 0x2f,
		0x2f, 0x2f, 0x2f, 0x2f, 0x2f, 0x2f, 0x2f, 0x2f,
	}
	digitRangeAfterData = [simdBlockSize]int8{
		0x3a, 0x3a, 0x3a, 0x3a, 0x3a, 0x3a, 0x3a, 0x3a,
		0x3a, 0x3a, 0x3a, 0x3a, 0x3a, 0x3a, 0x3a, 0x3a,
	}
	lowerLetterRangeBeforeData = [simdBlockSize]int8{
		0x60, 0x60, 0x60, 0x60, 0x60, 0x60, 0x60, 0x60,
		0x60, 0x60, 0x60, 0x60, 0x60, 0x60, 0x60, 0x60,
	}
	lowerLetterRangeAfterData = [simdBlockSize]int8{
		0x67, 0x67, 0x67, 0x67, 0x67, 0x67, 0x67, 0x67,
		0x67, 0x67, 0x67, 0x67, 0x67, 0x67, 0x67, 0x67,
	}
	letterNibbleOffsetData = [simdBlockSize]byte{
		9, 9, 9, 9, 9, 9, 9, 9,
		9, 9, 9, 9, 9, 9, 9, 9,
	}

	evenShuffleData = [simdBlockSize]int8{
		0, 2, 4, 6, 8, 10, 12, 14,
		0, 2, 4, 6, 8, 10, 12, 14,
	}

	oddShuffleData = [simdBlockSize]int8{
		1, 3, 5, 7, 9, 11, 13, 15,
		1, 3, 5, 7, 9, 11, 13, 15,
	}

	evenShuffleDataAVX2 = [simdBlockSizeAVX2]int8{
		0, 2, 4, 6, 8, 10, 12, 14,
		0, 2, 4, 6, 8, 10, 12, 14,
		0, 2, 4, 6, 8, 10, 12, 14,
		0, 2, 4, 6, 8, 10, 12, 14,
	}

	oddShuffleDataAVX2 = [simdBlockSizeAVX2]int8{
		1, 3, 5, 7, 9, 11, 13, 15,
		1, 3, 5, 7, 9, 11, 13, 15,
		1, 3, 5, 7, 9, 11, 13, 15,
		1, 3, 5, 7, 9, 11, 13, 15,
	}
)

// encodeSIMD encodes complete SIMD blocks and returns the number of source
// bytes consumed.
func encodeSIMD(dst, src []byte) (processed int) {
	if len(src) < simdBlockSize || !archsimd.X86.AVX() {
		return 0
	}

	usedAVX2 := false
	if archsimd.X86.AVX2() {
		for len(src) >= simdBlockSizeAVX2 {
			usedAVX2 = true
			encodedLo, encodedHi := encodeBlockAVX2(archsimd.LoadUint8x32(src))
			encodedLo.Store(dst[:simdBlockSizeAVX2])
			encodedHi.Store(dst[simdBlockSizeAVX2 : 2*simdBlockSizeAVX2])

			src = src[simdBlockSizeAVX2:]
			dst = dst[2*simdBlockSizeAVX2:]
			processed += simdBlockSizeAVX2
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

	if usedAVX2 {
		archsimd.ClearAVXUpperBits()
	}

	return processed
}

// encodeBlock encodes one vector into two consecutive vectors of hexadecimal
// digits.
func encodeBlock(x archsimd.Uint8x16) (encodedLo, encodedHi archsimd.Uint8x16) {
	hexTableVector := archsimd.LoadUint8x16Array(&hexTableData)
	lowNibbleMaskVector := archsimd.LoadUint8x16Array(&lowNibbleMaskData)
	lowNibbles := x.And(lowNibbleMaskVector).BitsToInt8()
	highNibbles := x.ReshapeToUint16s().ShiftAllRight(4).ReshapeToUint8s().And(lowNibbleMaskVector).BitsToInt8()

	lowDigits := hexTableVector.PermuteOrZero(lowNibbles)
	highDigits := hexTableVector.PermuteOrZero(highNibbles)

	return highDigits.InterleaveLo(lowDigits), highDigits.InterleaveHi(lowDigits)
}

// encodeBlockAVX2 encodes one AVX2 vector into two consecutive vectors of
// hexadecimal digits.
func encodeBlockAVX2(x archsimd.Uint8x32) (encodedLo, encodedHi archsimd.Uint8x32) {
	hexTableVector := archsimd.LoadUint8x32Array(&hexTableDataAVX2)
	lowNibbleMaskVector := archsimd.BroadcastUint8x32(lowNibbleMask)
	lowNibbles := x.And(lowNibbleMaskVector).BitsToInt8()
	highNibbles := x.ReshapeToUint16s().ShiftAllRight(4).ReshapeToUint8s().And(lowNibbleMaskVector).BitsToInt8()

	lowDigits := hexTableVector.PermuteOrZeroGrouped(lowNibbles)
	highDigits := hexTableVector.PermuteOrZeroGrouped(highNibbles)

	lowDigitsLo := lowDigits.GetLo()
	highDigitsLo := highDigits.GetLo()
	encodedLo = encodedLo.SetLo(highDigitsLo.InterleaveLo(lowDigitsLo))
	encodedLo = encodedLo.SetHi(highDigitsLo.InterleaveHi(lowDigitsLo))

	lowDigitsHi := lowDigits.GetHi()
	highDigitsHi := highDigits.GetHi()
	encodedHi = encodedHi.SetLo(highDigitsHi.InterleaveLo(lowDigitsHi))
	encodedHi = encodedHi.SetHi(highDigitsHi.InterleaveHi(lowDigitsHi))

	return encodedLo, encodedHi
}

// decodeSIMD decodes complete SIMD blocks and returns the number of source
// bytes consumed.
func decodeSIMD(dst, src []byte) (processed int) {
	if len(src) < simdBlockSize || !archsimd.X86.AVX() {
		return 0
	}

	usedAVX2 := false
	if archsimd.X86.AVX2() {
		for len(src) >= simdBlockSizeAVX2 {
			usedAVX2 = true
			if invalid := decodeBlockAVX2(archsimd.LoadUint8x32(src), dst); invalid {
				archsimd.ClearAVXUpperBits()
				return processed
			}
			src = src[simdBlockSizeAVX2:]
			dst = dst[simdBlockSizeAVX2/2:]
			processed += simdBlockSizeAVX2
		}
	}

	for len(src) >= simdBlockSize {
		if invalid := decodeBlock(archsimd.LoadUint8x16(src), dst); invalid {
			if usedAVX2 {
				archsimd.ClearAVXUpperBits()
			}
			return processed
		}
		src = src[simdBlockSize:]
		dst = dst[simdBlockSize/2:]
		processed += simdBlockSize
	}

	if usedAVX2 {
		archsimd.ClearAVXUpperBits()
	}
	return processed
}

// decodeBlock decodes one vector of hexadecimal digits into eight bytes.
// It returns whether the input contained an invalid byte.
func decodeBlock(input archsimd.Uint8x16, dst []byte) (invalid bool) {
	c := input.BitsToInt8()
	normalizeLetterValue := archsimd.LoadUint8x16Array(&normalizeLetterValueData)
	digitRangeBefore := archsimd.LoadInt8x16Array(&digitRangeBeforeData)
	digitRangeAfter := archsimd.LoadInt8x16Array(&digitRangeAfterData)
	lowerLetterRangeBefore := archsimd.LoadInt8x16Array(&lowerLetterRangeBeforeData)
	lowerLetterRangeAfter := archsimd.LoadInt8x16Array(&lowerLetterRangeAfterData)
	lower := input.Or(normalizeLetterValue).BitsToInt8() // 'A'...'F' -> 'a'...'f'.
	isDigit := c.Greater(digitRangeBefore).And(digitRangeAfter.Greater(c))
	isLetter := lower.Greater(lowerLetterRangeBefore).And(lowerLetterRangeAfter.Greater(lower))
	isValid := isDigit.Or(isLetter)

	validBits := isValid.ToInt8x16().ToBits().ReshapeToUint64s()
	if validBits.GetElem(0) != ^uint64(0) || validBits.GetElem(1) != ^uint64(0) {
		return true
	}

	lowNibbleMaskVector := archsimd.LoadUint8x16Array(&lowNibbleMaskData)
	letterNibbleOffsetVector := archsimd.LoadUint8x16Array(&letterNibbleOffsetData)
	nibble := input.And(lowNibbleMaskVector).
		Add(isLetter.ToInt8x16().ToBits().And(letterNibbleOffsetVector)).BitsToInt8()

	evenShuffle := archsimd.LoadInt8x16Array(&evenShuffleData)
	oddShuffle := archsimd.LoadInt8x16Array(&oddShuffleData)
	evenNibbles := nibble.PermuteOrZero(evenShuffle)
	oddNibbles := nibble.PermuteOrZero(oddShuffle)
	result := evenNibbles.ToBits().ReshapeToUint16s().ShiftAllLeft(4).Or(oddNibbles.ToBits().ReshapeToUint16s()).ReshapeToUint64s()
	byteorder.LEPutUint64(dst, result.GetElem(0))
	return false
}

// decodeBlockAVX2 decodes one AVX2 vector of hexadecimal digits into sixteen
// bytes. It returns whether the input contained an invalid byte.
func decodeBlockAVX2(input archsimd.Uint8x32, dst []byte) (invalid bool) {
	lower := input.Or(archsimd.BroadcastUint8x32(lowLetter)) // 'A'...'F' -> 'a'...'f'.
	isDigit := input.Sub(archsimd.BroadcastUint8x32(digitRangeStart)).
		Less(archsimd.BroadcastUint8x32(digitRangeSize))
	isLetter := lower.Sub(archsimd.BroadcastUint8x32(letterRangeStart)).
		Less(archsimd.BroadcastUint8x32(letterRangeSize))
	isValid := isDigit.Or(isLetter)

	validBits := isValid.ToInt8x32().ToBits().ReshapeToUint64s()
	validBitsLo := validBits.GetLo()
	validBitsHi := validBits.GetHi()
	if validBitsLo.GetElem(0) != ^uint64(0) || validBitsLo.GetElem(1) != ^uint64(0) || validBitsHi.GetElem(0) != ^uint64(0) || validBitsHi.GetElem(1) != ^uint64(0) {
		return true
	}

	nibble := input.And(archsimd.BroadcastUint8x32(lowNibbleMask)).
		Add(isLetter.ToInt8x32().ToBits().And(archsimd.BroadcastUint8x32(letterNibbleOffset))).BitsToInt8()

	evenShuffle := archsimd.LoadInt8x32Array(&evenShuffleDataAVX2)
	oddShuffle := archsimd.LoadInt8x32Array(&oddShuffleDataAVX2)
	evenNibbles := nibble.PermuteOrZeroGrouped(evenShuffle)
	oddNibbles := nibble.PermuteOrZeroGrouped(oddShuffle)
	result := evenNibbles.ToBits().ReshapeToUint16s().ShiftAllLeft(4).Or(oddNibbles.ToBits().ReshapeToUint16s()).ReshapeToUint64s()
	byteorder.LEPutUint64(dst, result.GetLo().GetElem(0))
	byteorder.LEPutUint64(dst[8:], result.GetHi().GetElem(0))
	return false
}
