// Copyright 2026 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

//go:build goexperiment.simd && arm64

package hex

import (
	"internal/byteorder"
	"simd/archsimd"
)

const simdBlockSize = 16

var hexTableVector = archsimd.LoadUint8x16([]byte(hextable))

const lowNibbleMask = 0x0f
const lowLetter = 0x20 // 'A'...'F' -> 'a'...'f'.
const digitRangeStart = byte('0')
const digitRangeSize = byte('9') - digitRangeStart + 1
const letterRangeStart = byte('a')
const letterRangeSize = byte('f') - letterRangeStart + 1
const letterNibbleOffset = 9

// encodeSIMD encodes complete SIMD blocks and returns the number of source
// bytes consumed.
func encodeSIMD(dst, src []byte) (processed int) {
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
	lowNibbles := x.And(archsimd.BroadcastUint8x16(lowNibbleMask))
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
	lower := input.Or(archsimd.BroadcastUint8x16(lowLetter)) // 'A'...'F' -> 'a'...'f'.
	isDigit := input.Sub(archsimd.BroadcastUint8x16(digitRangeStart)).
		Less(archsimd.BroadcastUint8x16(digitRangeSize))
	isLetter := lower.Sub(archsimd.BroadcastUint8x16(letterRangeStart)).
		Less(archsimd.BroadcastUint8x16(letterRangeSize))
	isValid := isDigit.Or(isLetter)

	if isValid.ToInt8x16().ToBits().ReduceMin() == 0 {
		return true
	}

	nibble := input.And(archsimd.BroadcastUint8x16(lowNibbleMask)).
		Add(isLetter.ToInt8x16().ToBits().And(archsimd.BroadcastUint8x16(letterNibbleOffset))).BitsToInt8()

	evenNibbles := nibble.ConcatEven(nibble)
	oddNibbles := nibble.ConcatOdd(nibble)
	result := evenNibbles.ShiftAllLeft(4).Or(oddNibbles).ToBits().ReshapeToUint64s()
	byteorder.LEPutUint64(dst, result.GetElem(0))
	return false
}
