// Copyright 2026 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

//go:build goexperiment.simd && arm64

package hex

import "simd/archsimd"

const simdBlockSize = 16

var hexTableVector = archsimd.LoadUint8x16([]byte(hextable))
var digitRangeStart = archsimd.BroadcastUint8x16(0x30)       // '0'
var digitRangeEnd = archsimd.BroadcastUint8x16(0x39)         // '9'
var upperLetterRangeStart = archsimd.BroadcastUint8x16(0x41) // 'A'
var upperLetterRangeEnd = archsimd.BroadcastUint8x16(0x46)   // 'F'
var lowerLetterRangeStart = archsimd.BroadcastUint8x16(0x61) // 'a'
var lowerLetterRangeEnd = archsimd.BroadcastUint8x16(0x66)   // 'f'
// 0x61...0x66 - 0x57 = 0x0a...0x0f
var lowerLetterDeltaForSub = archsimd.BroadcastUint8x16(0x57)
var normalizeLetterValue = archsimd.BroadcastUint8x16(0x20)

// encodeSIMD encodes complete SIMD blocks and returns the number of source
// bytes consumed.
func encodeSIMD(dst, src []byte) int {
	processed := 0
	for len(src) >= simdBlockSize {
		encodedLo, encodedHi := encodeBlock(archsimd.LoadUint8x16(src))
		encodedLo.Store(dst[:simdBlockSize])
		encodedHi.Store(dst[simdBlockSize : 2*simdBlockSize])

		src = src[simdBlockSize:]
		dst = dst[2*simdBlockSize:]
		processed += simdBlockSize
	}
	return processed
}

// encodeBlock encodes one vector into two consecutive vectors of hexadecimal
// digits.
func encodeBlock(x archsimd.Uint8x16) (encodedLo, encodedHi archsimd.Uint8x16) {
	nibbleMask := archsimd.BroadcastUint8x16(0x0f)
	lowNibbles := x.And(nibbleMask)
	highNibbles := x.ShiftAllRight(4)

	lowDigits := hexTableVector.LookupOrZero(lowNibbles)
	highDigits := hexTableVector.LookupOrZero(highNibbles)

	return highDigits.InterleaveLo(lowDigits), highDigits.InterleaveHi(lowDigits)
}

// decodeSIMD decodes complete SIMD blocks and returns the number of source
// bytes consumed.
func decodeSIMD(dst, src []byte) (processed int) {
	for len(src) >= simdBlockSize {
		if invalid := decodeBlock(archsimd.LoadUint8x16(src), dst); invalid {
			return processed
		}
		src = src[simdBlockSize:]
		dst = dst[simdBlockSize/2:]
		processed += simdBlockSize
	}
	return processed
}

// decodeBlock decodes one vector of hexadecimal digits into eight bytes.
// It returns whether the input contained an invalid byte.
func decodeBlock(input archsimd.Uint8x16, dst []byte) (invalid bool) {
	isDigit := input.GreaterEqual(digitRangeStart).
		And(digitRangeEnd.GreaterEqual(input))
	isUpper := input.GreaterEqual(upperLetterRangeStart).
		And(upperLetterRangeEnd.GreaterEqual(input))
	isLower := input.GreaterEqual(lowerLetterRangeStart).
		And(lowerLetterRangeEnd.GreaterEqual(input))
	isValid := isDigit.Or(isUpper.Or(isLower))

	if isValid.ToInt8x16().ToBits().ReduceMin() == 0 {
		var invalidArr [16]int8
		isValid.ToInt8x16().StoreArray(&invalidArr)
		for _, v := range invalidArr {
			if v == 0 {
				return true
			}
		}
	}

	lower := input.Or(normalizeLetterValue)           // 'A'...'F' -> 'a'...'f'.
	letterNibble := lower.Sub(lowerLetterDeltaForSub) // 'a'...'f' ->  10...15
	digitNibble := input.Sub(digitRangeStart)         //  30...39  ->  0...9

	digitMask := isDigit.ToInt8x16()
	letterMask := digitMask.Not()
	nibble := digitNibble.BitsToInt8().And(digitMask).
		Or(letterNibble.BitsToInt8().And(letterMask))

	evenNibbles := nibble.ConcatEven(nibble)
	oddNibbles := nibble.ConcatOdd(nibble)
	result := evenNibbles.ShiftAllLeft(4).Or(oddNibbles)
	var arr [16]uint8
	result.ToBits().StoreArray(&arr)
	copy(dst[:8], arr[:8])
	return false
}
