    .section .text,"ax",@progbits
    .assume ADL = 1

    .globl _token_dot_offset_u8_96
    .type _token_dot_offset_u8_96,@function
_token_dot_offset_u8_96:
    ld hl, -1
    call __frameset

    ld iy, (ix + 6)     ; row cursor
    ld de, (ix + 9)     ; state cursor
    ld (ix - 1), 96
    ld bc, 0
    ld hl, 0

.Ldot96_loop:
    ld a, (iy)
    inc iy
    ld c, a
    ld a, (de)
    inc de
    ld b, a
    mlt bc
    add hl, bc
    dec (ix - 1)
    jr nz, .Ldot96_loop

    ld bc, (ix + 12)    ; subtract state_sum << 7
    or a
    sbc hl, bc
    ld bc, (ix + 15)    ; add precomputed row correction
    add hl, bc

    ld sp, ix
    pop ix
    ret

    .size _token_dot_offset_u8_96, .-_token_dot_offset_u8_96

    .globl _token_dot_offset_u8_128
    .type _token_dot_offset_u8_128,@function
_token_dot_offset_u8_128:
    ld hl, -1
    call __frameset

    ld iy, (ix + 6)     ; row cursor
    ld de, (ix + 9)     ; state cursor
    ld (ix - 1), 128
    ld bc, 0
    ld hl, 0

.Ldot128_loop:
    ld a, (iy)
    inc iy
    ld c, a
    ld a, (de)
    inc de
    ld b, a
    mlt bc
    add hl, bc
    dec (ix - 1)
    jr nz, .Ldot128_loop

    ld bc, (ix + 12)    ; subtract state_sum << 7
    or a
    sbc hl, bc
    ld bc, (ix + 15)    ; add precomputed row correction
    add hl, bc

    ld sp, ix
    pop ix
    ret

    .size _token_dot_offset_u8_128, .-_token_dot_offset_u8_128
