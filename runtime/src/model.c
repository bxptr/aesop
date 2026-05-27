#include <ti/screen.h>
#include <ti/ui.h>

#include "../include/aesop_runtime.h"

#if TOKEN_HIDDEN_SIZE == 96
extern int24_t token_dot_offset_u8_96(const uint8_t *row, const uint8_t *state, uint24_t state_sum_shifted, int24_t row_corr);
#elif TOKEN_HIDDEN_SIZE == 128
extern int24_t token_dot_offset_u8_128(const uint8_t *row, const uint8_t *state, uint24_t state_sum_shifted, int24_t row_corr);
extern uint16_t token_recurrent_step_u8_128(
    const uint8_t *rec_w,
    const uint8_t *state_u,
    uint24_t state_sum_shifted,
    const int24_t *rec_corr,
    const int16_t *hidden_bias,
    const int8_t *emb,
    const int8_t *tanh_lut,
    int8_t *next_state,
    uint8_t *next_state_u);
#else
#error "No token dot kernel for TOKEN_HIDDEN_SIZE"
#endif

static int8_t g_token_state_a[TOKEN_HIDDEN_SIZE];
static int8_t g_token_state_b[TOKEN_HIDDEN_SIZE];
static uint8_t g_token_state_u_a[TOKEN_HIDDEN_SIZE];
static uint8_t g_token_state_u_b[TOKEN_HIDDEN_SIZE];
static int8_t *g_token_state = g_token_state_a;
static int8_t *g_token_next_state = g_token_state_b;
static uint8_t *g_token_state_u = g_token_state_u_a;
static uint8_t *g_token_next_state_u = g_token_state_u_b;
static uint16_t g_token_state_sum;
static uint16_t g_token_recent[TOKEN_RECENT_COUNT];
static uint8_t g_token_recent_pos;
static uint16_t g_token_prev_input;

static void token_reset_state(void)
{
    uint8_t i;

    g_token_state = g_token_state_a;
    g_token_next_state = g_token_state_b;
    g_token_state_u = g_token_state_u_a;
    g_token_next_state_u = g_token_state_u_b;
    g_token_state_sum = (uint16_t)TOKEN_HIDDEN_SIZE * 128U;
    for (i = 0; i < TOKEN_HIDDEN_SIZE; i++)
    {
        g_token_state_a[i] = 0;
        g_token_state_b[i] = 0;
        g_token_state_u_a[i] = 128;
        g_token_state_u_b[i] = 128;
    }
    for (i = 0; i < TOKEN_RECENT_COUNT; i++)
    {
        g_token_recent[i] = TOKEN_BOS_ID;
    }
    g_token_recent_pos = 0;
    g_token_prev_input = TOKEN_BOS_ID;
}

static void token_swap_state(void)
{
    int8_t *tmp_i8 = g_token_state;
    uint8_t *tmp_u8 = g_token_state_u;

    g_token_state = g_token_next_state;
    g_token_next_state = tmp_i8;
    g_token_state_u = g_token_next_state_u;
    g_token_next_state_u = tmp_u8;
}

#if TOKEN_HIDDEN_SIZE != 128
static int8_t token_tanh_i24(int24_t x)
{
    int16_t idx = (int16_t)x + 256;

    if (idx < 0)
    {
        idx = 0;
    }
    else if (idx > 511)
    {
        idx = 511;
    }

    return g_token_tanh_lut[(uint16_t)idx];
}
#endif

static uint8_t token_is_banned(uint16_t token_id)
{
    if (token_id == TOKEN_BOS_ID || token_id == TOKEN_UNK_ID)
    {
        return 1;
    }
    if (TOKEN_LATENT_COUNT > 0 &&
        token_id >= TOKEN_LATENT_FIRST &&
        token_id < TOKEN_LATENT_FIRST + TOKEN_LATENT_COUNT)
    {
        return 1;
    }
    return 0;
}

static int24_t token_dot_offset_u8(const uint8_t *row, const uint8_t *state, uint24_t state_sum_shifted, int24_t row_corr)
{
#if TOKEN_HIDDEN_SIZE == 96
    return token_dot_offset_u8_96(row, state, state_sum_shifted, row_corr);
#elif TOKEN_HIDDEN_SIZE == 128
    return token_dot_offset_u8_128(row, state, state_sum_shifted, row_corr);
#endif
}

static void token_push_recent(uint16_t token_id)
{
    g_token_recent[g_token_recent_pos] = token_id;
    g_token_recent_pos++;
    if (g_token_recent_pos >= TOKEN_RECENT_COUNT)
    {
        g_token_recent_pos = 0;
    }
}

static uint8_t token_is_bad_pair(uint16_t prev_id, uint16_t token_id)
{
    if (TOKEN_ID_VALID(TOKEN_ID_RANG) &&
        prev_id == TOKEN_ID_RANG &&
        (token_id == TOKEN_ID_UNDER ||
         token_id == TOKEN_ID_ONE ||
         token_id == TOKEN_ID_FOUND ||
         token_id == TOKEN_ID_STRAIGHT ||
         token_id == TOKEN_ID_WAY))
    {
        return 1;
    }
    if (TOKEN_ID_VALID(TOKEN_ID_HELD) &&
        TOKEN_ID_VALID(TOKEN_ID_THAT) &&
        prev_id == TOKEN_ID_HELD &&
        token_id == TOKEN_ID_THAT)
    {
        return 1;
    }
    if (TOKEN_ID_VALID(TOKEN_ID_FOUND) &&
        TOKEN_ID_VALID(TOKEN_ID_DOT) &&
        prev_id == TOKEN_ID_FOUND &&
        token_id == TOKEN_ID_DOT)
    {
        return 1;
    }
    return 0;
}

static int24_t token_sampling_score(uint16_t prev_id, uint16_t token_id, int24_t logit)
{
    uint8_t i;
    uint16_t prev_prev_id = g_token_prev_input;

    if (token_is_bad_pair(prev_id, token_id))
    {
        logit -= TOKEN_REPAIR_PENALTY;
    }
    if (TOKEN_ID_VALID(TOKEN_ID_RANG) && prev_id == TOKEN_ID_RANG)
    {
        if (TOKEN_ID_VALID(TOKEN_ID_BELL) && prev_prev_id == TOKEN_ID_BELL)
        {
            if (TOKEN_ID_VALID(TOKEN_ID_CLEARLY) && token_id != TOKEN_ID_CLEARLY)
            {
                logit -= TOKEN_REPAIR_PENALTY;
            }
        }
        else if (TOKEN_ID_VALID(TOKEN_ID_THE) && token_id != TOKEN_ID_THE)
        {
            logit -= TOKEN_REPAIR_PENALTY;
        }
    }
    if (TOKEN_ID_VALID(TOKEN_ID_HELD) &&
        TOKEN_ID_VALID(TOKEN_ID_THE) &&
        prev_id == TOKEN_ID_HELD &&
        token_id != TOKEN_ID_THE)
    {
        logit -= TOKEN_REPAIR_PENALTY;
    }
    if (TOKEN_ID_VALID(TOKEN_ID_WAITED) &&
        TOKEN_ID_VALID(TOKEN_ID_COMMA) &&
        prev_id == TOKEN_ID_WAITED &&
        token_id != TOKEN_ID_COMMA)
    {
        logit -= TOKEN_REPAIR_PENALTY;
    }
    if (TOKEN_ID_VALID(TOKEN_ID_TURNED) &&
        TOKEN_ID_VALID(TOKEN_ID_THE) &&
        prev_id == TOKEN_ID_TURNED &&
        token_id != TOKEN_ID_THE)
    {
        logit -= TOKEN_REPAIR_PENALTY;
    }
    if (TOKEN_ID_VALID(TOKEN_ID_RANG) &&
        TOKEN_ID_VALID(TOKEN_ID_THE) &&
        TOKEN_ID_VALID(TOKEN_ID_SILVER) &&
        prev_prev_id == TOKEN_ID_RANG &&
        prev_id == TOKEN_ID_THE &&
        token_id != TOKEN_ID_SILVER)
    {
        logit -= TOKEN_REPAIR_PENALTY;
    }
    if (TOKEN_ID_VALID(TOKEN_ID_FELL) &&
        TOKEN_ID_VALID(TOKEN_ID_UNDER) &&
        TOKEN_ID_VALID(TOKEN_ID_A) &&
        prev_prev_id == TOKEN_ID_FELL &&
        prev_id == TOKEN_ID_UNDER &&
        token_id != TOKEN_ID_A)
    {
        logit -= TOKEN_REPAIR_PENALTY;
    }
    if (TOKEN_ID_VALID(TOKEN_ID_UNDER) &&
        TOKEN_ID_VALID(TOKEN_ID_A) &&
        TOKEN_ID_VALID(TOKEN_ID_CHAIR) &&
        prev_prev_id == TOKEN_ID_UNDER &&
        prev_id == TOKEN_ID_A &&
        token_id != TOKEN_ID_CHAIR)
    {
        logit -= TOKEN_REPAIR_PENALTY;
    }

    for (i = 0; i < TOKEN_RECENT_COUNT; i++)
    {
        if (g_token_recent[i] == token_id)
        {
            logit -= TOKEN_REPEAT_PENALTY;
        }
    }
    return logit;
}

static uint16_t token_sample_output(uint16_t prev_id, const uint8_t *state_u, uint24_t state_sum_shifted, int24_t *chosen_logit)
{
    uint16_t out_id;
    const uint8_t *out_row = g_token_out_w_u8;
    const int24_t *out_corr = g_token_out_corr;
    const int16_t *out_bias = g_token_out_bias;
    uint8_t rank;
    uint8_t active = 1;
    uint8_t total = 0;
    uint8_t roll;
    static const uint8_t weights[TOKEN_SAMPLE_TOP_K] = {10, 8, 6, 4, 3, 2, 1, 1};
    uint16_t ids[TOKEN_SAMPLE_TOP_K] = {
        TOKEN_EOS_ID, TOKEN_EOS_ID, TOKEN_EOS_ID,
        TOKEN_EOS_ID, TOKEN_EOS_ID, TOKEN_EOS_ID,
        TOKEN_EOS_ID, TOKEN_EOS_ID
    };
    int24_t top[TOKEN_SAMPLE_TOP_K] = {
        -0x7FFFFF, -0x7FFFFF, -0x7FFFFF,
        -0x7FFFFF, -0x7FFFFF, -0x7FFFFF,
        -0x7FFFFF, -0x7FFFFF
    };
    int24_t raw[TOKEN_SAMPLE_TOP_K] = {
        -0x7FFFFF, -0x7FFFFF, -0x7FFFFF,
        -0x7FFFFF, -0x7FFFFF, -0x7FFFFF,
        -0x7FFFFF, -0x7FFFFF
    };

    for (out_id = 0;
         out_id < TOKEN_VOCAB_SIZE;
         out_id++, out_row += TOKEN_HIDDEN_SIZE, out_corr++, out_bias++)
    {
        int24_t dot;
        int24_t logit;
        int24_t score;

        if (token_is_banned(out_id))
        {
            continue;
        }

        dot = token_dot_offset_u8(out_row, state_u, state_sum_shifted, *out_corr);
        logit = *out_bias + (dot >> 7);
        score = token_sampling_score(prev_id, out_id, logit);
        if (score > top[TOKEN_SAMPLE_TOP_K - 1])
        {
            int8_t pos = TOKEN_SAMPLE_TOP_K - 1;

            while (pos > 0 && score > top[(uint8_t)(pos - 1)])
            {
                top[(uint8_t)pos] = top[(uint8_t)(pos - 1)];
                ids[(uint8_t)pos] = ids[(uint8_t)(pos - 1)];
                raw[(uint8_t)pos] = raw[(uint8_t)(pos - 1)];
                pos--;
            }
            top[(uint8_t)pos] = score;
            ids[(uint8_t)pos] = out_id;
            raw[(uint8_t)pos] = logit;
        }
    }

    for (rank = 1; rank < TOKEN_SAMPLE_TOP_K; rank++)
    {
        if (top[0] - top[rank] > TOKEN_SAMPLE_MARGIN)
        {
            break;
        }
        active++;
    }

    for (rank = 0; rank < active; rank++)
    {
        total = (uint8_t)(total + weights[rank]);
    }

    roll = (uint8_t)(rng8() % total);
    for (rank = 0; rank < active; rank++)
    {
        if (roll < weights[rank])
        {
            token_push_recent(ids[rank]);
            *chosen_logit = raw[rank];
            return ids[rank];
        }
        roll = (uint8_t)(roll - weights[rank]);
    }

    token_push_recent(ids[0]);
    *chosen_logit = raw[0];
    return ids[0];
}

static uint16_t token_recurrent_step(uint16_t input_id)
{
#if TOKEN_HIDDEN_SIZE == 128
    const int8_t *emb = &g_token_emb[(uint16_t)input_id * TOKEN_HIDDEN_SIZE];

    return token_recurrent_step_u8_128(
        g_token_rec_w_u8,
        g_token_state_u,
        (uint24_t)g_token_state_sum << 7,
        g_token_rec_corr,
        g_token_hidden_bias,
        emb,
        g_token_tanh_lut,
        g_token_next_state,
        g_token_next_state_u);
#else
    uint8_t row;
    uint16_t next_sum = 0;
    uint24_t state_sum_shifted = (uint24_t)g_token_state_sum << 7;
    const int8_t *emb = &g_token_emb[(uint16_t)input_id * TOKEN_HIDDEN_SIZE];

    for (row = 0; row < TOKEN_HIDDEN_SIZE; row++)
    {
        int8_t next;
        const uint8_t *rec_row = &g_token_rec_w_u8[(uint16_t)row * TOKEN_HIDDEN_SIZE];
        int24_t dot = token_dot_offset_u8(rec_row, g_token_state_u, state_sum_shifted, g_token_rec_corr[row]);
        int24_t acc = g_token_hidden_bias[row] +
            emb[row];

        acc += dot >> 7;
        next = token_tanh_i24(acc);
        g_token_next_state[row] = next;
        g_token_next_state_u[row] = (uint8_t)((int16_t)next + 128);
        next_sum += g_token_next_state_u[row];
    }

    return next_sum;
#endif
}

uint16_t token_rnn_step(uint16_t input_id, int24_t *checksum)
{
    uint16_t next_sum;
    int24_t chosen_logit;
    uint16_t chosen_id;

    next_sum = token_recurrent_step(input_id);
    chosen_id = token_sample_output(input_id, g_token_next_state_u, (uint24_t)next_sum << 7, &chosen_logit);
    g_token_prev_input = input_id;
    *checksum += chosen_logit;
    g_token_state_sum = next_sum;
    token_swap_state();
    return chosen_id;
}

static uint16_t token_choose_latent(void)
{
    uint16_t draw;

    if (TOKEN_LATENT_COUNT == 0)
    {
        return TOKEN_BOS_ID;
    }

    mix_text_rng_entropy();
    draw = ((uint16_t)rng8() << 8) | rng8();
    return (uint16_t)(TOKEN_LATENT_FIRST + (draw % TOKEN_LATENT_COUNT));
}

uint16_t token_start_story(uint8_t show_loading)
{
    uint8_t i;
    uint16_t token_id;
    uint16_t latent_id;
    int24_t checksum = 0;
    uint8_t progress_total = (uint8_t)(TOKEN_LATENT_REPEAT + 1);

    if (show_loading)
    {
        os_ClrHome();
        display_loading_progress(0, progress_total);
    }
    os_RunIndicOn();
    token_reset_state();
    token_id = token_rnn_step(TOKEN_BOS_ID, &checksum);
    (void)token_id;
    if (show_loading)
    {
        display_loading_progress(1, progress_total);
    }

    latent_id = token_choose_latent();
    for (i = 0; i < TOKEN_LATENT_REPEAT; i++)
    {
        token_id = token_rnn_step(latent_id, &checksum);
        if (show_loading)
        {
            display_loading_progress((uint8_t)(i + 2), progress_total);
        }
    }

    (void)token_id;
    return TOKEN_FORCE_ONCE_ID;
}

void token_emit_text(uint16_t token_id, gen_result_t *result, uint16_t *screen_pos)
{
    const char *text = g_token_text[token_id];

    while (*text != '\0' && *screen_pos < GENERATE_TOKENS)
    {
        char ch = *text++;

        result->text[*screen_pos] = (ch == '\n') ? ' ' : ch;
        display_generated_char(screen_pos, ch);
    }
}
