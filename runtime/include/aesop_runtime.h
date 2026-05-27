#pragma once

#include <stdint.h>

#include "../src/generated_model.h"

#if !defined(MODEL_ARCH_TOKEN_RNN) || !MODEL_ARCH_TOKEN_RNN
#error "AESOP runtime expects a generated token RNN model."
#endif

#define DISPLAY_COLS 26
#define DISPLAY_ROWS 8
#define GENERATE_TOKENS (DISPLAY_COLS * DISPLAY_ROWS)
#define PAGE_HISTORY_COUNT 8
#define TOKEN_RECENT_COUNT 8
#define TOKEN_SAMPLE_TOP_K 8
#define TOKEN_SAMPLE_MARGIN 64
#define TOKEN_REPEAT_PENALTY 96
#define TOKEN_REPAIR_PENALTY 4096
#define TOKEN_ID_VALID(id) ((id) != TOKEN_ID_INVALID)
#define GEN_APPVAR_NAME "GENOUT"
#define TEXT_RNG_APPVAR_NAME "GENRNG"

#ifndef TOKEN_LATENT_REPEAT
#define TOKEN_LATENT_REPEAT 4
#endif

typedef struct
{
    char magic[4];
    uint8_t version;
    uint8_t h;
    uint16_t vocab;
    uint16_t tokens;
    uint32_t cycles_generate;
    uint16_t last_id;
    char text[GENERATE_TOKENS + 1];
} __attribute__((packed)) gen_result_t;

typedef struct
{
    gen_result_t pages[PAGE_HISTORY_COUNT];
    uint8_t count;
    uint8_t pos;
} page_history_t;

void load_text_rng_state(void);
void save_text_rng_state(void);
uint8_t rng8(void);
void mix_text_rng_entropy(void);
void write_gen_results(const gen_result_t *result);

uint8_t wait_for_story_action(void);
uint8_t poll_exit_key(void);
void display_generated_char(uint16_t *screen_pos, char ch);
void display_loading_progress(uint8_t step, uint8_t total);
void page_history_reset(page_history_t *history);
void page_history_push(page_history_t *history, const gen_result_t *result);
uint8_t page_history_back(page_history_t *history, uint8_t *action);
uint8_t page_history_forward(page_history_t *history, uint8_t *action);

uint16_t token_start_story(uint8_t show_loading);
uint16_t token_rnn_step(uint16_t input_id, int24_t *checksum);
void token_emit_text(uint16_t token_id, gen_result_t *result, uint16_t *screen_pos);
