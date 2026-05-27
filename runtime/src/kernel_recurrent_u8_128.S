    .section .text,"ax",@progbits
    .assume ADL = 1

    .globl _token_recurrent_step_u8_128
    .type _token_recurrent_step_u8_128,@function
_token_recurrent_step_u8_128:
    ld hl, -32
    call __frameset

    ld hl, (ix + 6)     ; rec_w row cursor
    ld (ix - 5), hl
    ld hl, (ix + 9)     ; state_u base
    ld (ix - 26), hl
    ld hl, (ix + 12)    ; state_sum << 7
    ld (ix - 29), hl
    ld hl, (ix + 15)    ; rec_corr cursor
    ld (ix - 8), hl
    ld hl, (ix + 18)    ; hidden bias cursor
    ld (ix - 11), hl
    ld hl, (ix + 21)    ; embedding cursor for input token
    ld (ix - 14), hl
    ld hl, (ix + 24)    ; tanh LUT
    ld (ix - 32), hl
    ld hl, (ix + 27)    ; next_state cursor
    ld (ix - 17), hl
    ld hl, (ix + 30)    ; next_state_u cursor
    ld (ix - 20), hl

    ld (ix - 1), 128    ; output row count
    ld hl, 0
    ld (ix - 23), hl    ; uint16 next_sum

.Lrec128_row:
    ld iy, (ix - 5)     ; row cursor
    ld de, (ix - 26)    ; state cursor
    ld (ix - 2), 128
    ld bc, 0
    ld hl, 0

.Lrec128_dot:
    ld a, (iy)
    inc iy
    ld c, a
    ld a, (de)
    inc de
    ld b, a
    mlt bc
    add hl, bc
    dec (ix - 2)
    jr nz, .Lrec128_dot

    push iy             ; persist row cursor for next row
    pop de
    ld (ix - 5), de

    ld bc, (ix - 29)    ; subtract state_sum << 7
    or a
    sbc hl, bc
    ld iy, (ix - 8)     ; add row correction and advance corr cursor
    ld bc, (iy)
    add hl, bc
    lea iy, iy + 3
    push iy
    pop de
    ld (ix - 8), de

    ld c, 7             ; match the compiler's recurrent logical shift
    call __ishru

    push hl
    pop iy              ; IY = dot >> 7

    ld hl, (ix - 11)    ; add int16 hidden bias, advance by 2 bytes
    ld de, (hl)
    add.sis iy, de
    inc hl
    inc hl
    ld (ix - 11), hl

    ld hl, (ix - 14)    ; add sign-extended int8 embedding
    ld a, (hl)
    inc hl
    ld (ix - 14), hl
    ld l, a
    rlc l
    sbc.sis hl, hl
    ex.sis de, hl
    ld e, a
    add.sis iy, de

    ld.sis de, 256      ; LUT index = clamp(acc + 256, 0, 511)
    add.sis iy, de
    ld.sis de, 511
    ex de, hl
    ld e, iyl
    ld d, iyh
    ex de, hl
    or a, a
    sbc.sis hl, de
    ld e, iyl
    ld d, iyh
    jr c, .Lrec128_upper_ready
    ld.sis de, 511

.Lrec128_upper_ready:
    ex de, hl
    ld e, iyl
    ld d, iyh
    ex de, hl
    ld.sis bc, 0
    or a, a
    sbc.sis hl, bc
    call pe, __setflag
    ld l, c
    ld h, b
    jp m, .Lrec128_index_ready
    ex.sis de, hl

.Lrec128_index_ready:
    ld de, 0
    ld e, l
    ld d, h
    ld hl, (ix - 32)
    add hl, de
    ld c, (hl)          ; next signed state byte

    ld hl, (ix - 17)
    ld (hl), c
    inc hl
    ld (ix - 17), hl

    ld l, -128          ; uint8 view is int8 + 128 modulo 256
    ld a, c
    xor a, l
    ld e, a
    ld hl, (ix - 20)
    ld (hl), e
    inc hl
    ld (ix - 20), hl

    ld d, 0
    ld hl, (ix - 23)
    add.sis hl, de
    ld (ix - 23), hl

    dec (ix - 1)
    jp nz, .Lrec128_row

    ld hl, (ix - 23)
    ld sp, ix
    pop ix
    ret

    .size _token_recurrent_step_u8_128, .-_token_recurrent_step_u8_128
