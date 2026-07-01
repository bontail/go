// Copyright 2026 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

//go:build goexperiment.simd && wasm

package hex

import (
	"internal/byteorder"
	"simd/archsimd"
)

const simdBlockSize = 16

var hexTableVector = archsimd.LoadUint8x16([]byte(hextable)).BitsToInt8()
var decodePackMask = archsimd.LoadInt8x16([]int8{
	0, 2, 4, 6, 8, 10, 12, 14,
	-1, -1, -1, -1, -1, -1, -1, -1,
})

const lowNibbleMask = 0x0f
const lowByteMask = 0x00ff
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

	lowDigits := hexTableVector.LookupOrZero(lowNibbles.BitsToInt8()).ToBits()
	highDigits := hexTableVector.LookupOrZero(highNibbles.BitsToInt8()).ToBits()

	encodedLo = highDigits.ExtendLo8ToUint16().
		Or(lowDigits.ExtendLo8ToUint16().ShiftAllLeft(8)).
		ReshapeToUint8s()
	encodedHi = highDigits.ExtendHi8ToUint16().
		Or(lowDigits.ExtendHi8ToUint16().ShiftAllLeft(8)).
		ReshapeToUint8s()
	return encodedLo, encodedHi
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

	validBits := isValid.ToInt8x16().ToBits().ReshapeToUint64s()
	if validBits.GetElem(0) != ^uint64(0) || validBits.GetElem(1) != ^uint64(0) {
		return true
	}

	nibble := input.And(archsimd.BroadcastUint8x16(lowNibbleMask)).
		Add(isLetter.ToInt8x16().ToBits().And(archsimd.BroadcastUint8x16(letterNibbleOffset))).BitsToInt8()

	words := nibble.ToBits().ReshapeToUint16s()
	packed := words.And(archsimd.BroadcastUint16x8(lowByteMask)).ShiftAllLeft(4).
		Or(words.ShiftAllRight(8)).ReshapeToUint8s().BitsToInt8().
		LookupOrZero(decodePackMask).ToBits().ReshapeToUint64s()
	byteorder.LEPutUint64(dst, packed.GetElem(0))
	return false
}
